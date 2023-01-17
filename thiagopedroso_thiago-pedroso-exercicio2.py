import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dt_frame_bf = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

dt_frame_bf.info()
dt_frame_bf.head(20)
fig, ax = plt.subplots(figsize=(15,8))

age_value = dt_frame_bf.sort_values(by=["Age"])['Age']

sns.violinplot(ax=ax,x=age_value, y=dt_frame_bf["Purchase"], width=1, palette="Blues")#, bw=0.1

dt_frame_bf_prod_id = dt_frame_bf['Product_ID'].value_counts().rename_axis('Produto').reset_index(name='Qtd_Compras')

dt_frame_bf_prod_id = dt_frame_bf_prod_id.sort_values('Qtd_Compras', ascending=False).head(10)



fig, ax = plt.subplots(figsize=(15,8))



sns.barplot(x=dt_frame_bf_prod_id['Produto'], y=dt_frame_bf_prod_id['Qtd_Compras'], palette="deep", ax=ax)

filtro = dt_frame_bf['Occupation'].isin(dt_frame_bf['Occupation'].value_counts().head(5).index)



fig, ax = plt.subplots(figsize=(15,8))

sns.set_style("whitegrid")

v = sns.boxplot(x='Age', y='Purchase', data= dt_frame_bf[filtro],hue='Occupation',  palette="Blues")
sns.catplot(x = 'Marital_Status',

            y = 'Purchase',

            hue = 'Marital_Status',

            margin_titles = True,

            kind = 'violin',

            col = 'Occupation',

            data = dt_frame_bf[dt_frame_bf['Purchase'] > 9000],

            aspect = .7,

            col_wrap = 5)