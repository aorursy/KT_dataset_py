import math

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn import preprocessing, cross_validation



df_test = pd.read_csv('../input/test.csv', index_col = 'id')

df = pd.read_csv('../input/train.csv', index_col = 'id')

print(df.head())

#create new column in dataframe equal to species

df['label'] = df['species']



#create numpy array storing unique species values

species = df.species.unique() 



#replace newly created column in dataframe with index of each species in the array. 

for sp in species:

    df['label'].replace(sp, species.tolist().index(sp), inplace = True)

    

#store label column in variable y 

y= df['label']
print(df.describe())
scaled_X = preprocessing.scale(df.drop(['species','label'], 1))

scaled_X_test = preprocessing.scale(df_test)
pca = PCA()

pca_X = pca.fit_transform(scaled_X)

eigenvectors_ini = pca.components_                                      #dim = (m x 192)

eigenvalues_ini = pca.explained_variance_
pca = PCA(n_components = 120)

pca_X = pca.fit_transform(scaled_X)

eigenvectors = pca.components_                                          #dim = (m x 120)

eigenvalues = pca.explained_variance_



var_retained = np.sum(eigenvalues, axis =0)/np.sum(eigenvalues_ini, axis = 0)

print('Variance retained:', var_retained) 
pca_X_test = pca.fit_transform(scaled_X_test)
pca_X_train, pca_X_cv, y_train, y_cv = cross_validation.train_test_split(pca_X, y, test_size=0.2)

print('Length training data',len(pca_X_train), '\n', 'length cv data', len(pca_X_cv))
colors = 100*['k','gray','m','r','orange','y','g','c','b','w']



#Plot PCA'd data (reduced data)

fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

for i in range(len(df)):

    ax.scatter(pca_X[i,0],pca_X[i,1],pca_X[i,2], color = colors[df['label'].iloc[i]])

ax.set_xlabel('PCA Dimension 1')

ax.set_ylabel('PCA Dimension 2')

ax.set_zlabel('PCA Dimension 3')

ax.set_title('Leaf training data visualisation (PCA = 3)')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn import neighbors, svm

from sklearn.ensemble import RandomForestClassifier



#Logistic Regression

clf_log = LogisticRegression(multi_class='multinomial', solver = 'newton-cg')

clf_log.fit(pca_X_train, y_train)

accuracy_log = clf_log.score(pca_X_cv, y_cv)

print('Logistic Regression accuracy:',accuracy_log)



#K-Nearest Neighbours

clf_knn = neighbors.KNeighborsClassifier()

clf_knn.fit(pca_X_train, y_train)

accuracy_knn = clf_knn.score(pca_X_cv, y_cv)

print('K-Nearest Neighbours accuracy:',accuracy_knn)



#SVM 

clf_svm = svm.SVC(probability = True)

clf_svm.fit(pca_X_train, y_train)

accuracy_svm = clf_svm.score(pca_X_cv, y_cv)

print('SVM accuracy:',accuracy_svm)



#Random Forest

clf_rf = RandomForestClassifier(n_estimators=120)

clf_rf.fit(pca_X_train, y_train)

accuracy_rf = clf_rf.score(pca_X_cv, y_cv)

print('Random forest accuracy:',accuracy_rf)

def plot_accuracy(accuracy_log, accuracy_knn, accuracy_svm, accuracy_rf):

    fig, axes = plt.subplots()

    ax=axes

    colors = ['r','orange','y','g']

    width = 0.5

    x = range(0, 4)

    y = [accuracy_log, accuracy_knn, accuracy_svm, accuracy_rf]

    ind = ['Logistic Reg', 'K-Nearest Neighb', 'SVM', 'Random Forest']

    ax.bar(x, y, width, align = 'center', color = colors)

    for a, b in enumerate(y):

        ax.text(x[a] - width/4, b - 0.2, str(round(b,3)), color = 'k')

    ax.xaxis.set_ticks(x)

    ax.set_xticklabels(ind)

    ax.axhline(min(y), linewidth=1, color='b')

    ax.set_title('Accuracies - CrossValidation data')

    plt.show()



#Plot accuracy

plot_accuracy(accuracy_log, accuracy_knn, accuracy_svm, accuracy_rf)
#Logistic Regression

labels_log = clf_log.predict(pca_X_test)



#K-Nearest Neighbours

labels_knn = clf_knn.predict(pca_X_test)



#Random Forest

labels_rf = clf_rf.predict(pca_X_test)



#SVM - can't use predict(X) as in previous cases

prob_svm = clf_svm.predict_proba(pca_X_test)

labels_svm = prob_svm.argmax(axis = 1)
#Function to summarise predictions

def get_predictions_summary(species, labels_log, labels_knn, labels_svm, labels_rf, data):



    df_pred = pd.DataFrame(data = data)

    df_pred['logistic'] = species[labels_log]

    df_pred['k-nn'] = species[labels_knn]

    df_pred['svm'] = species[labels_svm]

    df_pred['random_forest'] = species[labels_rf]

    #df_pred.to_csv('All predictions.csv',sep=',')



    df_pred_summary = pd.DataFrame({'logistic': df_pred['logistic'].value_counts()})

    df_pred_summary['k-nn'] = pd.DataFrame(data = df_pred['k-nn'].value_counts())

    df_pred_summary['svm'] = pd.DataFrame(data = df_pred['svm'].value_counts())

    df_pred_summary['random_forest'] = pd.DataFrame(data = df_pred['random_forest'].value_counts())



    return df_pred, df_pred_summary



#Call function to summarise predictions

df_pred, df_pred_summary = get_predictions_summary(species, labels_log, labels_knn, labels_svm, labels_rf, df_test)





#Function to plot top N predictions per model

def plot_predictions_summary(df_pred, df_pred_summary, top_n):

    

    cols = int(len(df_pred_summary.columns)/2)

    rows =  len(df_pred_summary.columns) - cols

    colors = ['r','orange','y','g']

    

    fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (9,6))

    ax_0 = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]



    i = 0

    for name in df_pred.columns[-4:]:

        ax1 = df_pred_summary.sort_values(str(name), ascending = False)[str(name)].iloc[0:top_n].plot(ax = ax_0[i], kind = 'bar', color = colors[i])

        ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation=20, fontsize = 6)

        ax1.set_title(str(name))

        i+=1 



    for column in df_pred_summary.columns:

        print('Top', top_n, column, '\n', df_pred_summary.sort_values(str(column), ascending = False)[str(column)].iloc[0:top_n])

    

    plt.show()



#Call function to plot top-5 species

plot_predictions_summary(df_pred, df_pred_summary, 5)

#Nr of predictions that match across models (models predicting the same) & analyse concurrent predictions across models (in CV or test data).

def get_same_predictions_across_models(pred_log, pred_knn, pred_svm, pred_rf, *labels):



    #store 2-model matches

    matches_logvknn = matches_logvsvm = matches_logvrf = matches_knnvsvm = matches_knnvrf = matches_svmvrf = 0

    #store 2-model matches with labels - CV only

    ismatch2logvknn = ismatch2logvsvm = ismatch2logvrf = ismatch2knnvsvm = ismatch2knnvrf = ismatch2svmvrf = 0

    #store 2-by-2-model matches

    matches_2and2_logknn_svmrf = matches_2and2_logsvm_knnrf = matches_2and2_logrf_knnsvm = 0

    #store 3-model matches

    matches_logvknnvsvm = matches_logvknnvrf = matches_logvsvmvrf = matches_knnvsvmvrf = 0

    #store 3-model matches with labels - CV only

    isamatch3log = isamatch3knn = isamatch3svm = isamatch3rf = isamatchall4 = 0

    #store 4-model matches

    matches_all = 0



    for i in range(len(pred_log)):

        if (pred_log[i] == pred_knn[i]):    

            if (pred_log[i] == pred_svm[i]):                

                if (pred_log[i] == pred_rf[i]):             

                    matches_all += 1                            #match all

                    try:

                        if pred_log[i] == labels[0][i]:

                            isamatchall4 += 1                       #prediction match labels - CV only

                    except:

                        pass

                else:

                    matches_logvknnvsvm += 1                    #match 3 models (Log, Knn, SVM)

                    try:

                        if pred_log[i] == labels[0][i]:                   

                            isamatch3rf += 1                        #prediction match labels - CV only

                    except:

                        pass

            elif(pred_log[i] == pred_rf[i]):

                matches_logvknnvrf += 1

                try:

                    if pred_log[i] == labels[0][i]:                    

                        isamatch3svm += 1                           #prediction match labels - CV only

                except:

                    pass

                

            else:

                if (pred_svm[i] == pred_rf[i]):              

                    matches_2and2_logknn_svmrf += 1             #match 2 models (Log, Knn) and 2 models (SVM, Rf)

                else:

                    matches_logvknn += 1                        #match 2 models (Log, Knn)



        elif (pred_log[i] == pred_svm[i]):

            if(pred_svm[i] == pred_rf[i]):

                matches_logvsvmvrf += 1                         #match 3 models (Log, SVM, Rf)

                try:

                    if pred_log[i] == labels[0][i]:

                        isamatch3knn += 1                               #prediction match labels - CV only

                except:

                    pass

            else:

                if (pred_knn[i] == pred_rf[i]):

                    matches_2and2_logsvm_knnrf += 1             #match 2 models (Log, SVM) and 2 models (Knn, Rf)

                else:

                    matches_logvsvm += 1                        #match 2 models (Log, SVM)



        elif(pred_log[i] == pred_rf[i]):

            if (pred_knn[i] == pred_svm[i]):

                    matches_2and2_logrf_knnsvm += 1             #match 2 models (Log, Rf) and 2 models (Knn, SVM)

            else:

                matches_logvrf += 1                             #match 2 models (Log, Rf)

            

        elif (pred_knn[i] == pred_svm[i]):

            if (pred_knn[i] == pred_rf[i]):

                matches_knnvsvmvrf += 1                         #match 3 models (Knn, SVM, Rf)

                try:

                    if pred_knn[i] == labels[0][i]:

                        isamatch3log += 1                               #prediction match labels - CV only

                except:

                    pass

            else:

                matches_knnvsvm += 1                            #match 2 models (Knn, SVM)

        elif(pred_knn[i] == pred_rf[i]):

            matches_knnvrf += 1                                 #match 2 models (Knn, Rf)



            

        elif (pred_svm[i] == pred_rf[i]):

            matches_svmvrf += 1                                 #match 2 models (SVM, Rf)





    matches3 = [matches_knnvsvmvrf, matches_logvsvmvrf, matches_logvknnvrf, matches_logvknnvsvm]

    matches2and2 = [matches_2and2_logknn_svmrf, matches_2and2_logsvm_knnrf, matches_2and2_logrf_knnsvm]

    matches2 = [matches_logvknn, matches_logvsvm, matches_logvrf, matches_knnvsvm, matches_knnvrf, matches_svmvrf]

    actualmatches = [isamatch3log, isamatch3knn, isamatch3svm, isamatch3rf,

                     ismatch2logvknn, ismatch2logvsvm, ismatch2logvrf, ismatch2knnvsvm, ismatch2knnvrf, ismatch2svmvrf]

    tags3match = ['(no Log)','(no KNN)','(no SVM)','(no Rf)']

    tags2match = ['Log-Knn','Log-SVM','Log-Rf','Knn-SVM','Knn-Rf','SVM-Rf']



    print('All 4 model match',matches_all)

    print('3 model match:')

    for i in range(len(matches3)):

        print(tags3match[i], matches3[i],'perc:',matches3[i]/len(pred_log))

    print('2-and-2 model match:')

    for i in range(len(matches2and2)):

        print(tags2match[i], tags2match[len(matches2)-1-i], matches2and2[i],'perc:',matches2and2[i]/len(pred_log))

    print('2 model match:')

    for i in range(len(matches2)):

        print(tags2match[i], matches2[i],'perc:',matches2[i]/len(pred_log))

    print('Actual matches - 3 model (CV set only):')

    for i in range(len(matches3)):

        print(tags3match[i], actualmatches[i],'perc:',actualmatches[i]/len(pred_log))

    print('Actual matches - 2 model (CV set only):')

    for i in range(len(matches2)):

        print(tags2match[i], actualmatches[i+4],'perc:',actualmatches[i+4]/len(pred_log))



    return matches_all, isamatchall4, matches3, matches2and2, matches2, actualmatches



#Call function to compare predictions across models - CV or Test data

matches_all, isamatchall4, matches3, \

matches2and2, matches2, actualmatches = get_same_predictions_across_models(labels_log, labels_knn, labels_svm, labels_rf)



#Function to plot matches

def plot_matches_across_models(matches3, matches2and2, matches2, actualmatches):

    y_ini = [matches3, matches2and2, matches2]                                                                      #matches in predictions across models

    ind_ini = [['-Log', '-KNN', '-SVM', '-R.For'], ['LgKn-SVRf', 'LgSV-KnRf', 'LgRf-KnSV'],

           ['LgKn','LgSV','LgRf','KnSV','KnRf','SVRf']]                                              #tags for 3-model matches, 2-and-2 model matches & 2-model matches  

    z_ini = [actualmatches[0:4], [0,0,0], actualmatches[4:10]]                                                      #matches across models and with labels - CV data only

    

    cols = len(y_ini)

    y = []

    x = []

    x_2 = []

    idx = []

    z = []

    ind = []



    for match in y_ini:                                                                                             #populate data to plot for elements of y_ini different from zero 

        if all(v == 0 for v in match):

            cols -= 1     

        else:

            index = y_ini.index(match)

            idx.append(index)

            y.append(match)

            x.append(range(0,len(match)))

            x_2.append(np.arange(0.5, (len(match)+0.5)))

            z.append(z_ini[index])

            ind.append(ind_ini[index])     

                

    fig, axes = plt.subplots(nrows = 1, ncols = cols, figsize = (9,6))

    colors = ['r','orange','y','g']

    width = 0.5



    for i in range(len(y)):

        axes[i].bar(x[i], y[i], width, align = 'center', color = colors)

        for a, b in enumerate(y[i]):

            axes[i].text(x[i][a] - width/4, b + b*0.02, str(round(b,3)), color = 'k')

        if not (all(v == 0 for v in z[i])):                                                                         #only cv data (when there are labels to match predictions with)

            axes[i].bar(x_2[i], z[i], width, align = 'center', color = 'c')

            for a, b in enumerate(z[i]):

                axes[i].text(x_2[i][a] - width/4, b + b*0.02, str(round(b,3)), color = 'k')

        axes[i].xaxis.set_ticks(x[i])

        axes[i].set_xticklabels(ind[i])

        axes[i].tick_params(labelsize = 'small')

        axes[i].set_title(str(math.ceil((idx[i]+3)/(idx[i]+1))) + ' models match', fontsize= 8)

    plt.show()



#Call function to plot matching predictions between models

plot_matches_across_models(matches3, matches2and2, matches2, actualmatches)

def get_predictions_probabilities(species, df_data, pca_data):



    df_prob = pd.DataFrame(data = np.nan, index = df_data.index, columns = [str(sp) for sp in species])



    #Logistic Regression

    prob_log = clf_log.predict_proba(pca_data)

    for sp in species:

        df_prob[str(sp)] = prob_log[:,species.tolist().index(sp)]



    df_prob = df_prob[sorted(df_prob.columns)]

    print(df_prob.tail())

    df_prob.to_csv('submission.csv',sep=',')



    return df_prob



df_prob = get_predictions_probabilities(species, df_test, pca_X_test)
