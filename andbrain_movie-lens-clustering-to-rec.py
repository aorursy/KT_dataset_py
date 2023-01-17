
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/ml100k/resultados.csv', index_col=0)

%matplotlib inline
import matplotlib.pyplot as plt
ap = df.loc['AP']

def f(x):
    return x[4] + ' - ' + x[5]
ap_experimentos = ap
ap_experimentos['Experimentos'] = ap_experimentos.apply(f,axis=1)
ap_experimentos = ap_experimentos.set_index('Experimentos')

ax = ap_experimentos[['RMSE', 'MAE']].plot.bar(
    figsize=(12,5),
    fontsize=16,
    ylim=[0.8,1.4]
)
ax.set_title("Resultados do AP Variando Valores da Diagonal Principal", fontsize=20)
# import seaborn as sns
# sns.despine(bottom=True, left=True)
ap_clusters = ap
ap_clusters = ap_clusters.set_index(['Description','Method'])
ap_clusters[['Clusters', 'RMSE', 'MAE', 'Predictions']]
ap_average = ap.set_index(['Method'])
average = ap_average.loc['Average']
average = average.set_index('Description')
# average[['RMSE']]
axes = average[['RMSE']].plot.bar(ylim=[1,1.13])
plt.show()
kmeans = df.loc['Kmeans']
kmeans_clusters = kmeans.set_index(['Clusters','Method'])
kmeans_clusters[['RMSE', 'MAE', 'Predictions']]
kmeans_average = kmeans.set_index(['Method'])
average = kmeans_average.loc['Average']
average = average.set_index('Clusters')
# average[['RMSE']]
axes = average[['RMSE']].plot.bar(ylim=[1,1.1])
plt.show()
sd = df.loc['Multi Kmeans']
sd_clusters = sd.set_index(['Clusters','Method'])
sd_clusters[['RMSE', 'MAE', 'Predictions']]
sd_average = sd.set_index(['Method'])
average = sd_average.loc['Average']
average = average.set_index('Clusters')
# average[['RMSE']]
axes = average[['RMSE']].plot.bar(ylim=[1,1.1])
plt.show()
df_ml100k_tt = pd.read_csv('../input/ml100ktraintest/treino_teste.csv')
import seaborn as sns
df_ml100k_tt
box = df_ml100k_tt[['set','1','2','3','4','5']]
box = box[(box.set == 'u1')]
sns.boxplot(
data=box
)
df_ml100k_exp2 = pd.read_csv('../input/ml100kexp2/resultados1.csv')
df_CV = df_ml100k_exp2[(df_ml100k_exp2.Algorithm != 'Multi Kmeans')]
df_exp2 = df_CV[['Referencia', 'Algorithm', 'Clusters', 'Method', 'RMSE']]
df_exp2.set_index(['Referencia', 'Algorithm', 'Method', 'Clusters'])
df_exp2

count = [0] * 21
i = 0
it = 0
for index, row in df_exp2.iterrows():
    if(i % 21 == 0):
        it = 0
#     print(i, it, index, row['RMSE'])
    count[it] = count[it] + row['RMSE']
    i = i + 1
    it = it + 1
# print(count)
finalList = [j/5 for j in count]
print(finalList)
