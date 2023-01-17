import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("ggplot")



import plotly.graph_objects as go



from IPython.display import clear_output

import warnings

warnings.filterwarnings("ignore")



data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.drop(['Unnamed: 32','id'], axis = 1, inplace=True)

data.head()
radius_mean_m = data[data['diagnosis'] =='M']['radius_mean']

radius_mean_b = data[data['diagnosis'] =='B']['radius_mean']



fig = go.Figure()

fig.add_trace(go.Histogram(x=radius_mean_b, name = 'Benign'))

fig.add_trace(go.Histogram(x=radius_mean_m, name = 'Malignant'))



# Overlay both histograms

fig.update_layout(title = 'Histogram Comparison of radius_mean', 

                  title_x = 0.5,

                  xaxis_title ='Radius Mean Value ->',

                  yaxis_title = 'Value Counts ->',

                  barmode='overlay')



# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()

print(f'Mean of radius_mean values (Benign): {radius_mean_b.mean()}')

print(f'Mean of radius_mean values (Malignant): {radius_mean_m.mean()}')
radius_mean_m = data[data['diagnosis'] =='M']['radius_mean']

radius_mean_b = data[data['diagnosis'] =='B']['radius_mean']



# Calculating Qurtile Points

desc = radius_mean_b.describe()

print(desc)

Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR



# Finding Outliers

a = radius_mean_b[radius_mean_b < lower_bound].values 

b = radius_mean_b[radius_mean_b > upper_bound].values

outliers = np.concatenate([a,b], axis = 0)



print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")

print(f'Outliers: {outliers}')
radius_mean_m = data[data['diagnosis'] =='M']['radius_mean']

radius_mean_b = data[data['diagnosis'] =='B']['radius_mean']



fig = go.Figure()

fig.add_trace(go.Box(y=radius_mean_m, name='Malignant', marker_color = 'indianred'))

fig.add_trace(go.Box(y=radius_mean_b, name = 'Benign', marker_color = 'lightseagreen'))



fig.update_layout(title='Distribution of radius_mean for Benign and Malignant Class',

                  title_x = 0.5,

                  xaxis_title = 'Feature',

                  yaxis_title = 'Value',

                  height = 400,

                  width = 800)

fig.show()
data.describe()
def ecdf(x):

    x = np.sort(x)

    def result(v):

        return np.searchsorted(x, v, side='right') / x.size

    return result



fig = go.Figure()

fig.add_scatter(x=np.unique(radius_mean_b), 

                y=ecdf(radius_mean_b)(np.unique(radius_mean_b)), 

                line_shape='hv')



fig.update_layout(title='CDF Curve for the feature radius_mean', title_x = 0.5,   

                  xaxis_title = 'Radius Mean Value',

                  yaxis_title = 'CDF',

                  height = 400, width = 600)

fig.show()
plt.figure(figsize = (15,8))

sns.jointplot(x = data['radius_mean'], y= data['area_mean'] ,kind="reg", color='green')

plt.title('Relation between radius_mean and area_mean')

plt.grid()

plt.show()



plt.figure(figsize = (15,8))

sns.jointplot(x = data['radius_mean'], y= data['texture_mean'] ,kind="reg", color='crimson')

plt.title('Relation between radius_mean and texture_mean')

plt.grid()

plt.show()
# Also we can look relationship between more than 2 distribution

sns.set(style = "darkgrid")

# df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se" ]]

df = data[['diagnosis', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'radius_mean']]

g = sns.PairGrid(df, hue = 'diagnosis',diag_sharey = False)

g.map_upper(sns.regplot )

g.map_lower(sns.kdeplot, color = 'blue')

g.map_diag(sns.kdeplot, color = 'green')

plt.show()
df = data[['diagnosis', 'smoothness_mean','compactness_mean',	'concavity_mean']]

g = sns.pairplot(df, hue = 'diagnosis', )

g.map_lower(sns.kdeplot, color = 'blue')

g.map_upper(sns.regplot)
f,ax=plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(method='pearson'),annot= True,linewidths=0.6,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Pearson Correlation Map')

plt.savefig('graph.png')

plt.show()
f,ax=plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(method='spearman'),annot= True,linewidths=0.6,fmt = ".1f",cmap="YlGnBu_r", ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Pearson Correlation Map')

plt.savefig('graph.png')

plt.show()
salary = [1,4,3,2,5,4,2,3,1,500]

print("Mean of salary: ",np.mean(salary))
print("Median of salary: ",np.median(salary))
# parameters of normal distribution

mu, sigma = 110, 20  # mean and standard deviation

s = np.random.normal(mu, sigma, 100000)

print("mean: ", np.mean(s))

print("standart deviation: ", np.std(s))

# visualize with histogram

plt.figure(figsize = (10,7))

plt.hist(s, 100, )

plt.ylabel("frequency")

plt.xlabel("IQ")

plt.title("Histogram of IQ")

plt.show()