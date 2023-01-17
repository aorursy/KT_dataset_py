from collections import defaultdict
from datetime import date, datetime, timedelta
from glob import glob
import math
from pprint import pprint
import re

from dateutil.relativedelta import relativedelta
from dateutil import parser as dateutil_parser
import numpy as np
import pandas as pd


pd.set_option('display.max_colwidth', None)


INPUT_CVM_DATA_PATH = '/kaggle/input/brazilian-investment-funds-data'
RELEVANT_FUNDS_COLUMNS = ['fund_name', 'class', 'start_date_in_class', 'anbima_class', 'baseline', 'long_term',
                          'start_date', 'min_application', 'shares_qty', 'net_worth', 'entry_fee', 'exit_fee']
FOREIGN_TERMS = r'AMERICANO|BDR|S&P|EXTERIOR|EXTERNA|CAMBIO|CÂMBIO|CAMBIAL|GLOBAL|GLOBAIS|DOLAR|DÓLAR|WESTERN'


DEFAULT_LIMIT_DATE = date(2020, 8, 14)
input_limit_date = input(f'Last available funds performance date (default={DEFAULT_LIMIT_DATE.isoformat()}): ')
limit_date = dateutil_parser.parse(input_limit_date) if input_limit_date else DEFAULT_LIMIT_DATE


consider_foreign_investments = input(f'Consider foreign investments (default=False): ')
consider_foreign_investments = consider_foreign_investments.lower() == 'true' if consider_foreign_investments else False
# UTILS FUNCTIONS

def _get_daily_info_df(filename, usecols=[0, 1, 3], names=['fund_id', 'date', 'quota'], parse_dates=[1]):
    return pd.read_csv(
        filename,
        sep=';', 
        usecols=usecols, 
        parse_dates=parse_dates,
        names=names,
        skiprows=1, 
    )


def _get_daily_info_file_fqn_from_date(date_):
    filename = f'inf_diario_fi_{date_.year}{str(date_.month).zfill(2)}.csv'
    return f'{INPUT_CVM_DATA_PATH}/{filename}'


def get_maisretorno_funds_comparison_link(ranking_df, qty=15):
    link = 'https://maisretorno.com/comparacao/principal/otimo/cdi,ibov/'
    
    for cnpj in ranking_df.index[0:qty]:
        link = f'{link}{re.sub("[^0-9]+", "", cnpj)},'
    
    return link[:-1]
    

def get_available_cvm_dates():
    """Return a list containing all the CVM workdays ordered."""
    available_cvm_dates = []
    for file_fqn in glob(f'{INPUT_CVM_DATA_PATH}/inf_diario_fi_*.csv'):
        df = _get_daily_info_df(file_fqn, usecols=[0, 1], names=['fund_id', 'date'])
        available_cvm_dates += set(df['date'].values)

    return sorted(available_cvm_dates)


AVAILABLE_CVM_DATES = get_available_cvm_dates()


def get_closest_existing_cvm_date(reference_date):
    """Return the same date if it exists in CVM, otherwise return the previous existing one.

    If neither the same date, nor a previous date exists, return the first date available.
    """
    reference_date = pd.Timestamp(reference_date)
    
    try:
        AVAILABLE_CVM_DATES.index(reference_date)
    
    except ValueError:
        reverse_ordered_available_dates = sorted(AVAILABLE_CVM_DATES, reverse=True)
        reference_date = next(iter(d for d in reverse_ordered_available_dates if d < reference_date), 
                              AVAILABLE_CVM_DATES[0])

    return pd.Timestamp(reference_date)
# BUILD SAMPLINGS PERIODS

def build_sampling_periods(qty_years_to_start, period_length, slide_length, last_date=limit_date):
    """Yield period tuples, making sure all dates exist in CVM calendar.
    
    E.g.: >>> list(build_sampling_periods(3, relativedelta(years=1), relativedelta(months=6)), date(2020, 8, 14))
    [(pandas.Timestamp('2017-08-14 00:00:00'), pandas.Timestamp('2018-08-14 00:00:00')), 
     (pandas.Timestamp('2018-02-14 00:00:00'), pandas.Timestamp('2019-02-14 00:00:00')), 
     (pandas.Timestamp('2018-08-14 00:00:00'), pandas.Timestamp('2019-08-14 00:00:00')), 
     (pandas.Timestamp('2019-02-14 00:00:00'), pandas.Timestamp('2020-02-14 00:00:00')), 
     (pandas.Timestamp('2019-08-14 00:00:00'), pandas.Timestamp('2020-08-14 00:00:00'))]
    """
    reference_date = last_date - relativedelta(years=qty_years_to_start)
    
    slice_begin_date = get_closest_existing_cvm_date(reference_date)
    slice_end_date = get_closest_existing_cvm_date(reference_date + period_length)

    while slice_end_date <= last_date:
        yield slice_begin_date, slice_end_date

        reference_date += slide_length
        
        slice_begin_date = get_closest_existing_cvm_date(reference_date)
        slice_end_date = get_closest_existing_cvm_date(reference_date + period_length)


period_length = relativedelta(years=1)
slide_length = relativedelta(months=1)
PERIOD_SLICES_10Y = list(build_sampling_periods(10, period_length=period_length, slide_length=slide_length))
PERIOD_SLICES_7Y = list(build_sampling_periods(7, period_length=period_length, slide_length=slide_length))
PERIOD_SLICES_5Y = list(build_sampling_periods(5, period_length=period_length, slide_length=slide_length))
PERIOD_SLICES_4Y = list(build_sampling_periods(4, period_length=period_length, slide_length=slide_length))
PERIOD_SLICES_3Y = list(build_sampling_periods(3, period_length=period_length, slide_length=slide_length))


pprint(PERIOD_SLICES_3Y)
# BUILD FUNDS METADATA 

# PART 1: load essential data into {df_funds_meta}
df_funds_meta = pd.read_csv(
    f'{INPUT_CVM_DATA_PATH}/cad_fi.csv', 
    sep=';', 
    encoding='cp1252',
    index_col=0,
    usecols=[0, 1, 7, 10, 11, 12, 13, 15, 16, 17],
    names=['fund_id', 'fund_name', 'start_date', 'class', 'start_date_in_class', 
           'baseline', 'condominium_type', 'exclusive', 'long_term', 'qualified'],
    skiprows=1, 
)

df_funds_meta_compl = pd.read_csv(
    f'{INPUT_CVM_DATA_PATH}/extrato_fi.csv', 
    sep=';', 
    encoding='cp1252',
    index_col=0, 
    usecols=[0, 9, 10, 17, 37, 40], 
    names=['fund_id', 'anbima', 'anbima_class', 'min_application', 'entry_fee', 'exit_fee'],
    skiprows=1, 
)

df_funds_meta[['anbima', 'anbima_class', 'min_application', 'entry_fee', 'exit_fee']] = df_funds_meta_compl[['anbima', 'anbima_class', 'min_application', 'entry_fee', 'exit_fee']]

# # PART 2: applied_value
# # TODO: need to remove data of funds which start on CVM after a pre-existing active period
# df = pd.DataFrame()
# current_date = date(2005, 1, 1)
# while current_date <= DEFAULT_LIMIT_DATE:
#     current_file_fqn = _get_daily_info_file_fqn_from_date(current_date)
#     df = pd.concat([df, _get_daily_info_df(current_file_fqn, usecols=[0, 1, 5, 6], names=['fund_id', 'date', 'raising', 'withdraw'])], ignore_index=True)
#     current_date += relativedelta(months=1)
# df = df[df['date'] <= pd.Timestamp(limit_date.isoformat())].groupby('fund_id').sum()
# df_funds_meta['applied_value'] = df['raising'] - df['withdraw']

# PART 3: net_worth, shares_qty
youngest_funds_daily_df = _get_daily_info_df(filename=_get_daily_info_file_fqn_from_date(limit_date),
                                             usecols=[0, 1, 4, 7], 
                                             names=('fund_id', 'date', 'net_worth', 'shares_qty'))
youngest_funds_daily_df = youngest_funds_daily_df[youngest_funds_daily_df['date'] == pd.Timestamp(limit_date.isoformat())]
youngest_funds_daily_df = youngest_funds_daily_df.set_index('fund_id', verify_integrity=True)
assert not youngest_funds_daily_df.isnull().values.any()
df_funds_meta[['net_worth', 'shares_qty']] = youngest_funds_daily_df[['net_worth', 'shares_qty']]

# # PART 4: net_worth_vs_applied_value_factor
# TODO: need the part 2 fixing
# df_funds_meta['net_worth_vs_applied_value_factor'] = df_funds_meta['net_worth'] / df_funds_meta['applied_value'] 

# PART 5: filter to keep only relevant funds
df_funds_meta = df_funds_meta[
    (df_funds_meta['condominium_type'] == 'Aberto') &
    (df_funds_meta['exclusive'] == 'N') & 
    (df_funds_meta['qualified'] == 'N') &
    (df_funds_meta['anbima'] == 'S') &
#     (df_funds_meta['min_application'] <= 200000) &  # commented because this data is often outdated on CVM
    (df_funds_meta['shares_qty'] >= 50)
]

if not consider_foreign_investments:
    df_funds_meta = df_funds_meta[
        ~(df_funds_meta['fund_name'].str.contains(FOREIGN_TERMS, re.IGNORECASE)) &
        ~(df_funds_meta['anbima_class'].str.contains(FOREIGN_TERMS, re.IGNORECASE))
    ]

df_funds_meta = df_funds_meta[RELEVANT_FUNDS_COLUMNS]
df_funds_meta.head()
# MORE UTILS FUNCTIONS

# TODO: Result could be cached to improve performance
def build_funds_performance_df(start_date, end_date):
    """Build a DataFrame with the funds performance on the period informed via parameters."""
    first_month_df = _get_daily_info_df(_get_daily_info_file_fqn_from_date(start_date)).set_index('fund_id')
    last_month_df = _get_daily_info_df(_get_daily_info_file_fqn_from_date(end_date)).set_index('fund_id')
    
    first_day_df = first_month_df[first_month_df['date'] == start_date]
    last_day_df = last_month_df[last_month_df['date'] == end_date]
    
    assert not first_day_df.index.duplicated().any()
    assert not last_day_df.index.duplicated().any()

    funds_performance_df = last_day_df - first_day_df
    funds_performance_df['quota'] = funds_performance_df['quota'] / first_day_df['quota']
    funds_performance_df.rename(columns={'quota': 'accrued_quota', 'date': 'length'}, inplace=True)
    funds_performance_df = funds_performance_df.dropna().sort_values('accrued_quota', ascending=False)
    
    funds_performance_df['fund_name'] = df_funds_meta['fund_name']
    funds_performance_df.dropna(subset=['fund_name'], inplace=True)

    count = len(funds_performance_df)
    funds_performance_df['position'] = range(1, count + 1)

    assert not funds_performance_df.isnull().values.any()
    return funds_performance_df


def build_sampling_ranking_df(sampling_period_slices):
    """Rank the better funds considering their performances in each period slice from sampling_period_slices.

    :param sampling_period_slices: List of tuples, containing the date edges for each period slice of the sampling.
                                   E.g.: [(date(2019, 1, 1), date(2019, 12, 1)), (date(2020, 1, 1), date(2020, 12, 1))]

    :rtype: pandas.DataFrame
    :return: Ranking DataFrame
    """
    ranking_df = build_funds_performance_df(*sampling_period_slices[0])
    for period_slice in sampling_period_slices[1:]:
        ranking_df['position'] += build_funds_performance_df(*period_slice)['position']

    ranking_df.rename(columns={'position': 'sum_positions'}, inplace=True)
    ranking_df.drop(columns=['accrued_quota', 'length'], inplace=True)

    ranking_df['mean_positions'] = ranking_df['sum_positions'] / len(sampling_period_slices)
    ranking_df.sort_values('sum_positions', inplace=True)
    ranking_df['accrued_position'] = range(1, ranking_df.count()[0] + 1)

    return ranking_df
ranking_df_10 = build_sampling_ranking_df(PERIOD_SLICES_10Y)
ranking_df_10.head(n=30)
ranking_df_7 = build_sampling_ranking_df(PERIOD_SLICES_7Y)
ranking_df_7.head(n=30)
ranking_df_5 = build_sampling_ranking_df(PERIOD_SLICES_5Y)
ranking_df_5.head(n=30)
ranking_df_4 = build_sampling_ranking_df(PERIOD_SLICES_4Y)
ranking_df_4.head(n=30)
ranking_df_3 = build_sampling_ranking_df(PERIOD_SLICES_3Y)
ranking_df_3.head(n=30)
# PICKING UP THE TOP {TOP_QTY_TO_CONSIDER} FUNDS INSIDE RANKINGS AND EVALUATING THEIR PERFORMANCES IN ALL AVAILABLE RANKINGS

TOP_QTY_TO_CONSIDER = 50

ordered_rankings = [ranking_df_3, ranking_df_4, ranking_df_5, ranking_df_7, ranking_df_10]

# funds existing exclusively in these rankings' top 50 will NOT be considered
rankings_only_for_evaluation = [id(ranking_df_7), id(ranking_df_10)]

funds_positions = defaultdict(lambda: [99999] * len(ordered_rankings))
frequency_inside_top_qty = defaultdict(lambda: 0)

for _, ranking_to_pick_up in enumerate(ordered_rankings):
    if id(ranking_to_pick_up) in rankings_only_for_evaluation:
        continue
    
    for _, fund in ranking_to_pick_up[0:TOP_QTY_TO_CONSIDER].iterrows():
        fund_cnpj = fund.name
        
        if frequency_inside_top_qty[fund_cnpj]:
            continue  # fund already evaluated

        for idx, ranking_to_evaluate in enumerate(ordered_rankings):
            try:
                accrued_position = ranking_to_evaluate.loc[fund_cnpj]['accrued_position']
                funds_positions[fund_cnpj][idx] = accrued_position
                
                if accrued_position <= TOP_QTY_TO_CONSIDER:
                    frequency_inside_top_qty[fund_cnpj] += 1

            except KeyError:
                pass
# ORDERING BY RELEVANCE

funds_per_frequency = {frequency: [] for frequency in range(1, len(ordered_rankings) + 1)}
for fund_cnpj, frequency in sorted(frequency_inside_top_qty.items(), key=lambda x: x[1], reverse=True):
    funds_per_frequency[frequency].append((fund_cnpj, funds_positions[fund_cnpj]))

ordered_funds = []
for frequency, funds_infos in sorted(funds_per_frequency.items(), key=lambda x: x[0], reverse=True):
    for data in sorted(funds_infos, key=lambda x: (x[1][0], x[1][1], x[1][2])):
        ordered_funds.append((frequency, ) + data)
# BUILDING ACCRUED QUOTA PER PERIOD, AND RELATED STATS

def build_df_accrued_quota(start_date, end_date):
    first_month_df = _get_daily_info_df(_get_daily_info_file_fqn_from_date(start_date)).set_index('fund_id')
    last_month_df = _get_daily_info_df(_get_daily_info_file_fqn_from_date(end_date)).set_index('fund_id')
    
    first_day_df = first_month_df[first_month_df['date'] == start_date]
    last_day_df = last_month_df[last_month_df['date'] == end_date]
    
    assert not first_day_df.index.duplicated().any()
    assert not last_day_df.index.duplicated().any()

    funds_performance_df = (
        first_day_df
        .rename(columns={'quota': 'before'})
        .merge(last_day_df.rename(columns={'quota': 'after'}), left_index=True, right_index=True)
    )
    funds_performance_df = (
        funds_performance_df
        .assign(accrued_quota=(funds_performance_df['after'] - funds_performance_df['before']) / funds_performance_df['before'])
        .dropna()
        .sort_values('accrued_quota', ascending=False)
        [['accrued_quota']]
    )

    assert not funds_performance_df.isnull().values.any()
    return funds_performance_df


df_accrued_quotas = pd.DataFrame(index=df_funds_meta.index)
for idx, period in enumerate(PERIOD_SLICES_5Y):
    df_accrued_quotas[str(idx)] = build_df_accrued_quota(*period)['accrued_quota']
df_accrued_quotas_transposed = df_accrued_quotas.transpose()

for idx, data in enumerate(ordered_funds):
    frequency, fund_cnpj, positions = data
    
    values = list(filter(lambda x: not np.isnan(x), df_accrued_quotas_transposed[fund_cnpj].values))
    values = list(map(lambda x: round(x * 100, 2), values))
    sorted_values = sorted(values)
    len_ = len(values)
    
    fund_stats = {'accrued_quotas': {'values': values,
                                     'len': len_,
                                     'lowest': sorted_values[:2],
                                     'highest': sorted_values[-2:],
                                     'average': round(np.average(values), 2),
                                     'perc_below_5': round(len(list(filter(lambda x: x < 5, values))) / len_ * 100, 2),
                                     'perc_over_15': round(len(list(filter(lambda x: x > 15, values))) / len_ * 100, 2)}}
    
    ordered_funds[idx] = (frequency, fund_cnpj, positions, fund_stats)
# REMOVING FUNDS WITH PERFORMANCES WORSE THAN 5% IN MORE THAN 10% OF THE PERIODS

ordered_funds = list(filter(lambda x: x[3]['accrued_quotas']['perc_below_5'] <= 10, ordered_funds))
# GENERATING .csv FILE

MILLNAMES = ['',' k',' mi',' bi',' tri']


# thanks to https://stackoverflow.com/users/212538/janus for https://stackoverflow.com/a/3155023/10311090
def millify(n):
    """Humanize large numbers."""
    n = float(n)
    millidx = max(0,min(len(MILLNAMES)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), MILLNAMES[millidx])


def _treat_date(str_date):
    """Return a brazilian format string."""
    return dateutil_parser.parse(str_date).strftime('%d/%m/%Y')


csv_file = u'CNPJ;NOME;INI ATIV;#PER;2 MENORES;<5%;2 MAIORES;>15%;MÉDIA;RENTABILIDADES;CLASSE;DT INI CLASSE;ANBIMA;REF.;'\
           u'LONGO PRAZO;TX ADESÃO;TX RESGATE;PATRIMÔNIO;#COTISTAS;%;DISTRIBUIDORA;APL MÍNIMA;COTZ RESG;LIQ RESG;VOLATILIDADE;SHARPE\n'

for _, fund_cnpj, _, stats in ordered_funds:
    fund_meta = df_funds_meta.loc[fund_cnpj]
 
    fund_name = fund_meta['fund_name']
    start_date = _treat_date(fund_meta['start_date'])
    fund_class = fund_meta['class']
    start_date_in_class = _treat_date(fund_meta['start_date_in_class'])
    anbima_class = fund_meta['anbima_class']
    baseline = fund_meta['baseline']
    long_term = fund_meta['long_term']
    entry_fee = fund_meta['entry_fee']
    exit_fee = fund_meta['exit_fee']
    net_worth = millify(fund_meta['net_worth'])
    shares_qty = millify(fund_meta['shares_qty'])
    net_worth_x_shares_qty_factor = millify(fund_meta['net_worth'] / fund_meta['shares_qty'])
    stats = stats['accrued_quotas']

    # data below must be searched manually, because the related CVM data is often outdated
    distributor = 'PESQUISAR'
    min_application_value = 'PESQUISAR'
    days_to_convert = 'PESQUISAR'
    days_to_withdraw = 'PESQUISAR'
    volatility = 'PESQUISAR'
    sharpe = 'PESQUISAR'
 
    csv_file += f'{fund_cnpj};{fund_name};{start_date};'\
                f'{stats["len"]};{stats["lowest"]};{stats["perc_below_5"]}%;{stats["highest"]};{stats["perc_over_15"]}%;{stats["average"]}%;{stats["values"]};'\
                f'{fund_class};{start_date_in_class};{anbima_class};'\
                f'{baseline};{long_term};{entry_fee};{exit_fee};{net_worth};{shares_qty};{net_worth_x_shares_qty_factor};'\
                f'{distributor};{min_application_value};{days_to_convert};{days_to_withdraw};{volatility};{sharpe}\n'.replace('99999', '').replace('nan', '-').replace('.00;', ';')

with open('best_ranked_funds.csv', 'w') as f:
    f.write(csv_file)
    
print('CSV file best_ranked_funds.csv generated. Check `data>output`. Please set only semicolon (`;`) as delimiter.')