import pandas as pd
dataset = pd.read_csv('../input/data.csv')
dataset.head()
dataset['difference'] = (dataset['Potential'] - (dataset['Overall']))
def evolution(d):

    if d == 0:

        return "Stable"

    elif d >=1 and d<=5:

        return "Small"

    elif d >=6 and d<=10:

        return "Medium"

    elif d >11:

        return "Big"
dataset['Evolution'] = dataset['difference'].apply(evolution)
dataset.loc[(dataset['Evolution']== 'Big') & (dataset['Potential']>80)].sort_values(by='Potential', ascending=False)