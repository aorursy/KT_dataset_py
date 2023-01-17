import pandas as pd
df = pd.read_csv('../input/transactions.csv')
df['date_time'] = pd.to_datetime(df.timestamp * 1000000)
df.head(3)
# the original pizza wallet
BASE_SEEDS = ['1XPTgDRhN8RFnzniWCddobD9iKZatrvH4']
SATOSHI_PER_BTC = 10**7
df.head(3)
def dig_row(row, seeds, transactions, min_satoshis, trace_from_key):
    if row['satoshis'] < min_satoshis:
        return None
    trace_columns = {True: 'input_key', False: 'output_key'}
    if row[trace_columns[trace_from_key]] not in seeds:
        return None
    seeds.add(row['output_key'])
    transactions.append(row)


def single_pass_dig(initial_seeds, input_df, initial_datetime=None, min_satoshis=0, trace_from_key=True):
    df = input_df.copy()
    active_seeds = {i for i in initial_seeds}
    if trace_from_key and initial_datetime is not None:
        df = df[df['date_time'] >= initial_datetime]
    elif not(trace_from_key) and initial_datetime is not None:
        df = df[df['date_time'] <= initial_datetime]
    df.sort_values(by=['timestamp'], ascending=trace_from_key, inplace=True)
    transactions = []
    df.apply(lambda row: dig_row(row, active_seeds, transactions, min_satoshis, trace_from_key), axis=1)
    return pd.DataFrame(transactions)
# setting seed date to +/- 1 day from the actual pizza transaction to avoid
# worrying about when in the day the pizza was purchased
future_transactions = single_pass_dig(BASE_SEEDS, df, initial_datetime=pd.to_datetime("May 16, 2010"))
past_transactions = single_pass_dig(BASE_SEEDS, df, initial_datetime=pd.to_datetime("May 18, 2010"), trace_from_key=False)
total_flows = future_transactions[['input_key', 'output_key', 'satoshis']].groupby(
    by=['input_key', 'output_key']).sum().reset_index()
total_flows.head(3)
total_flows.to_csv('total_flows.csv', index=False)
total_flows.info()