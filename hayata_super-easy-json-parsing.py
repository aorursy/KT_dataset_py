import pandas as pd 
import json


def separate_json(series: pd.Series) -> pd.DataFrame():
    """
    
    Args:
        series: Series before json parsing 

    Returns: DataFrame

    """
    # TODO: Write TypeException
    
    if isinstance(series[0], str):
        return pd.DataFrame(json.loads(s) for s in series)
    return pd.DataFrame(s for s in series)

df = pd.read_csv('../input/train.csv', engine='python')
json_col = ['device', 'geoNetwork', 'totals', 'trafficSource']
nest_json_col = ['adwordsClickInfo']

df = df.join(separate_json(df[col_name]) for col_name in json_col).drop(json_col, axis=1)
df = df.join(separate_json(df[nest_json_col[0]])).drop(nest_json_col, axis=1)
df.head()
df.columns