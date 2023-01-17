import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('../input/cement2/qu.xlsx')
df['es']=df['Cw']/100*(df['e/Aw'])
df['est']=df['Cw']/100*np.log(df['e/Aw'])



df1=df.drop(["location",'Cw'],axis=1)
cor=df1.corr()
plt.subplots(figsize=(10,10))         
sns.heatmap(cor,square=True,annot=True)

g = sns.PairGrid(df, y_vars=["q"], x_vars=["e/Aw", "Aw",'es','est'], height=4.5, hue="location", aspect=1.1)
ax = g.map(plt.scatter, alpha=0.6)
ax = g.add_legend()

g = sns.PairGrid(df, y_vars=["Af"], x_vars=["e/Aw", "Aw",'es','est'], height=4.5, hue="location", aspect=1.1)
ax = g.map(plt.scatter, alpha=0.8)
ax = g.add_legend()

sns.pairplot(df,palette="husl",diag_kind="kde")



