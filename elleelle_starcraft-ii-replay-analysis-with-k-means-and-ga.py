import numpy as np

import pandas as pd



from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import homogeneity_score



kf = KFold(n_splits = 10) 

kmeans = KMeans(n_clusters = 7, random_state = 31133113)



starcraft = pd.read_csv('../input/starcraft.csv')

star = starcraft.loc[starcraft['TotalHours'].notnull()]

y = pd.DataFrame(star, columns = ['LeagueIndex'])-1 # make zero-indexed



best_attr = [['UniqueHotkeys', 'ComplexUnitsMade', 'ComplexAbilityUsed', 'MaxTimeStamp'],

             ['MinimapAttacks', 'ComplexUnitsMade', 'ComplexAbilityUsed', 'MaxTimeStamp'],

             ['APM', 'UniqueHotkeys', 'TotalMapExplored', 'UniqueUnitsMade', 'ComplexUnitsMade',

              'ComplexAbilityUsed', 'MaxTimeStamp'],

             ['UniqueHotkeys', 'MinimapAttacks', 'ActionLatency', 'WorkersMade', 

              'ComplexUnitsMade']]
for attr in best_attr:

    X = pd.DataFrame(star, columns = attr)

    X += 0.0000001

    X = X.apply(np.log)



    X_sample, X_validation, y_sample, y_validation = train_test_split(

        X, y, test_size=0.2, random_state = 13)

    

    sil_min = []

    sil_mean = []

    jaccard = []

    purity = []



    for train, test in kf.split(X_sample):        

        labels = kmeans.fit_predict(X_sample.iloc[train,:])

        sil_vals = silhouette_samples(X_sample.iloc[train,:], labels)

        sil_min.append(min(sil_vals))

        sil_mean.append(np.mean(sil_vals))

        

        jaccard.append(jaccard_similarity_score(y_sample.iloc[train,:], labels)) 

        purity.append(homogeneity_score(y_sample.iloc[train,:].values.flatten(), labels))

        

    print(attr)

    print('Avg Silhouette min: ' + str(np.mean(np.asarray(sil_min))))

    print('Avg Silhouette mean: ' + str(np.mean(np.asarray(sil_mean))))

    print('Avg Jaccarad siilarity: ' + str(np.mean(np.asarray(jaccard))))

    print('Avg Purity: ' + str(np.mean(np.asarray(purity))))

    print()
top_attr = ['UniqueHotkeys', 'ComplexUnitsMade', 'ComplexAbilityUsed', 'MaxTimeStamp']

X = pd.DataFrame(star, columns = top_attr)

X += 0.0000001

X = X.apply(np.log)



X_sample, X_validation, y_sample, y_validation = train_test_split(

    X, y, test_size=0.2, random_state = 13)

    

centers = kmeans.fit(X_sample)

labels = kmeans.predict(X_validation)

sil_vals = silhouette_samples(X_validation, labels)





print("Validation Silhouette min: " + str(min(sil_vals)))

print("Validation Silhouette mean: " + str(np.mean(sil_vals)))

star_centers = []



for i in range(7):

    star_centers.append(np.exp(centers.cluster_centers_[i]))

    

level_centers = pd.DataFrame(star_centers, columns = top_attr)

level_centers.index = range(1, len(level_centers)+1)

level_centers[level_centers <= 0.0000001] = 0



print(level_centers)
top_attr = ['APM', 'UniqueHotkeys', 'TotalMapExplored', 'UniqueUnitsMade', 'ComplexUnitsMade',

              'ComplexAbilityUsed', 'MaxTimeStamp']

X = pd.DataFrame(star, columns = top_attr)

X += 0.0000001

X = X.apply(np.log)



X_sample, X_validation, y_sample, y_validation = train_test_split(

    X, y, test_size=0.2, random_state = 13)

    

centers = kmeans.fit(X_sample)

labels = kmeans.predict(X_validation)

sil_vals = silhouette_samples(X_validation, labels)

star_centers = []



for i in range(7):

    star_centers.append(np.exp(centers.cluster_centers_[i]))

    

level_centers = pd.DataFrame(star_centers, columns = top_attr)

level_centers.index = range(1, len(level_centers)+1)

level_centers[level_centers <= 0.0000001] = 0



print("Validation Silhouette min: " + str(min(sil_vals)))

print("Validation Silhouette mean: " + str(np.mean(sil_vals)))

print()

print(level_centers)
import seaborn as sns



sns.boxplot(x = star['LeagueIndex'], y = star['APM'])
sns.boxplot(x = star['LeagueIndex'], y = star['UniqueHotkeys'])
sns.boxplot(x = star['LeagueIndex'], y = star['ComplexUnitsMade'])
sns.boxplot(x = star['LeagueIndex'], y = star['MaxTimeStamp'])