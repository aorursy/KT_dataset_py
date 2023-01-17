import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv('../input/TopStaredRepositories.csv')

df.head(20)
test = df['Number of Stars'].str.split("k")

test[0][0]
type(test)
test_df = df.copy()

#     test_df['Number of Stars'] = test_df['Number of Stars'].str.split("k")[0][0]
for index,row in df.iterrows():

    temp_number = df['Number of Stars'].str.split("k")[index][0]

    

# print(type(temp_number))

    df['Number of Stars'][index] = float(temp_number)*1000

    

#     print(temp_number)

    

#     if row['Number of Stars'].str.contains("k") == True:

#         print("k")

#     print(row['Number of Stars'])

#     print("1")
df.head()
clean_df=df.dropna(subset=["Language","Description"])
# pd.value_counts(clean_df['Language'].values, sort = True)
clean_df.isnull().sum()
clean_df.head()
clean_df.dtypes
sns.countplot(x="Language", data=clean_df, order=pd.value_counts(clean_df['Language']).iloc[:10].index)
def value_to_float(x):

    if type(x) == float or type(x) == int:

        return x

    if 'K' in x:

        if len(x) > 1:

            return float(x.replace('K', '')) * 1000

        return 1000.0

    return 0.0
clean_df.head()
wordcloud2 = WordCloud().generate(''.join(clean_df["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
js = clean_df.groupby('Language').get_group("JavaScript")

jv = clean_df.groupby('Language').get_group("Java")

py = clean_df.groupby('Language').get_group("Python")

rb = clean_df.groupby('Language').get_group("Ruby")

oc = clean_df.groupby('Language').get_group("Objective-C")

go = clean_df.groupby('Language').get_group("Go")

ht = clean_df.groupby('Language').get_group("HTML")

cs = clean_df.groupby('Language').get_group("CSS")

sw = clean_df.groupby('Language').get_group("Swift")

cp = clean_df.groupby('Language').get_group("C++")
js.head()
def popForLang(lang):

    return clean_df[clean_df['Language'] == lang]
wordcloud2 = WordCloud().generate(''.join(js["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('JavaScript')
wordcloud2 = WordCloud().generate(''.join(jv["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('Java')
wordcloud2 = WordCloud().generate(''.join(py["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('Python')
wordcloud2 = WordCloud().generate(''.join(rb["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('Ruby')
wordcloud2 = WordCloud().generate(''.join(oc["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('Objective-C')
wordcloud2 = WordCloud().generate(''.join(go["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('Go')
wordcloud2 = WordCloud().generate(''.join(ht["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('HTML')
wordcloud2 = WordCloud().generate(''.join(cs["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('CSS')
wordcloud2 = WordCloud().generate(''.join(sw["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('Swift')
wordcloud2 = WordCloud().generate(''.join(cp["Description"]))

# Generate plot

plt.imshow(wordcloud2)

plt.axis("off")

plt.show()
popForLang('C++')
# sns.barplot(x=clean_df["Language"], y=clean_df["Number of Stars"], data=clean_df, order=pd.value_counts(clean_df['Language']).iloc[:10].index)
kmean_df = clean_df.copy()
kmean_df.head()
pd.value_counts(kmean_df['Language'].values, sort = True)
le_encoders = {}

le = LabelEncoder()

le_encoders["Language"] = le

kmean_df["Language"] = le.fit_transform(kmean_df["Language"])
pd.value_counts(kmean_df['Language'].values, sort = True)
kmean_df.head()
kmean_df['Number of Stars']=kmean_df['Number of Stars'].astype("float64")

kmean_df['Language']=kmean_df['Language'].astype("float64")
kmean_df.dtypes
X = kmean_df.drop(['Username','Repository Name','Description','Last Update Date','Tags','Url'], axis=1)

X
X_scale = StandardScaler().fit_transform(X)
ks = []

costs = []

for k in range(2, 13):

    kmeans = KMeans(n_clusters=k).fit(X_scale)

    ks.append(k)

    costs.append(kmeans.inertia_)
sns.set_style("whitegrid",{'grid.linestyle':'--'})
sns.lineplot(x=ks, y=costs)
kmeans = KMeans(n_clusters=5).fit(X)
km = pd.DataFrame(kmeans.labels_, columns=['cluster'])

df_km = pd.concat([kmean_df, km], axis=1)
df_km.head()
x = df_km.drop(['Username','Repository Name','Description','Last Update Date','Tags','Url'], axis=1)
x = x.dropna()
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['xaxis', 'yaxis'])
finalDf = pd.concat([principalDf, km], axis = 1)
finalDf.head()
pd.value_counts(finalDf['cluster'].values, sort = True)
sns.scatterplot(y='xaxis', x='yaxis', hue='cluster', data=finalDf)
df_cluster0 = df_km[df_km['cluster'] == 0]
df_cluster0
df_cluster1 = df_km[df_km['cluster'] == 1]
df_cluster1
df_cluster2 = df_km[df_km['cluster'] == 2]
df_cluster2
df_cluster3 = df_km[df_km['cluster'] == 3]
df_cluster3
df_cluster4 = df_km[df_km['cluster'] == 4]
df_cluster4
sns.scatterplot(y='Language', x='Number of Stars', hue='cluster', data=df_km)
sns.scatterplot(y='Number of Stars', x='Language', hue='cluster', data=df_km)
sns.scatterplot(y='cluster', x='Number of Stars', hue='cluster', data=df_km)
sns.scatterplot(y='cluster', x='Language', hue='cluster', data=df_km)
sns.scatterplot(y='Number of Stars', x='cluster', hue='cluster', data=df_km)
sns.scatterplot(y='Number of Stars', x='Language', hue='cluster', data=df_km)
sns.scatterplot(y='Language', x='cluster', hue='cluster', data=df_km)
sns.scatterplot(y='cluster', x='Language', hue='cluster', data=df_km)