import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import warnings 

warnings.filterwarnings('ignore')



#Read the 4 csv files in separate dataframes

df_coh1=pd.read_csv('../input/cohort-1.csv')

df_coh2=pd.read_csv("../input/cohort-2.csv")

df_coh3=pd.read_csv("../input/cohort-3.csv")

df_coh4=pd.read_csv("../input/cohort-4.csv")

df_taxonomy=pd.read_csv("../input/Species Taxonomy.csv")

df_taxonomy.head(5)



#df_coh1.describe()

#df_coh1.info()
####################################################################

#First clean the datasets

####################################################################



def clean_data(df):

    # Remove NAN entries

    df.dropna()

    #### Transform categorical features into integers. One-hot-encoding not 

    #### needed because the features are binary. 

    # Define CRC sample=1, normal_control=0; male =0, female=1

    df_diag_series=df.grouped_diagnosis.map(lambda x: 0 if x=='normal_control' else 1)

    df_gender_series=df.gender.map(lambda x: 0 if x=='male' else 1)

    df_encoded = df.assign(diagnosis_int=pd.Series(df_diag_series.values))

    df_encoded = df_encoded.assign(gender_int=pd.Series(df_gender_series.values))

    #extract column order

    cols=df_encoded.columns.tolist()

    #rearrange columns in dataframe

    cols = cols[0:1]+cols[-2:]+cols[2:4]+cols[5:-2]

    df_encoded = df_encoded[cols]

    return df_encoded



df1 = clean_data(df_coh1)

df2 = clean_data(df_coh2)

df3 = clean_data(df_coh3)

df4 = clean_data(df_coh4)



#Check dataset

df3.head(3)

####################################################################

#Perform initial exploratory analysis

####################################################################

# Let's make some plots to see correlations, trends, clustering etc



import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import matplotlib.pylab as pylab



#set global matplotlib parameters

params = {'legend.fontsize': '16',

         'axes.labelsize': '16',

         'axes.titlesize':'18',

         'xtick.labelsize':'13',

         'font.weight' : 'medium',

         'ytick.labelsize':'13',

         'legend.fontsize': '14'}

pylab.rcParams.update(params)



# Plot 2D distribution of cancerous and control samples

# in age-BMI phase space



fig, ax = plt.subplots(2, 2, figsize=(10,10))

im=ax[0,0].scatter(df1.age, df1.BMI, c=df1.diagnosis_int, cmap='RdYlGn', label='Cohort 1')

#cbar = fig.colorbar(im, ticks=[0, 1])

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cancer',

                          markerfacecolor='g', markersize=10),

                   Line2D([0], [0], marker='o', color='w', label='Control',

                          markerfacecolor='r', markersize=10)

                  ]

ax[1,0].legend(handles=legend_elements, loc='upper left')



ax[0,1].scatter(df2.age, df2.BMI, c=df2.diagnosis_int, cmap='RdYlGn')

ax[1,0].scatter(df3.age, df3.BMI, c=df3.diagnosis_int, cmap='RdYlGn')

ax[1,1].scatter(df4.age, df4.BMI, c=df4.diagnosis_int, cmap='RdYlGn')

ax[0,0].set_ylabel('BMI')

ax[1,0].set_ylabel('BMI')

ax[1,0].set_xlabel('Age')

ax[1,1].set_xlabel('Age')

ax[0,0].set_title('Cohort 1' )

ax[0,1].set_title('Cohort 2')

ax[1,0].set_title('Cohort 3')

ax[1,1].set_title('Cohort 4')



# I don't see any particular clustering difference between cancer & non-cancer patients.



# Compare normalized age distribution of the 4 cohorts

fig, ax = plt.subplots(figsize=(10,5))

plt.hist(df1.age, color='white', edgecolor='hotpink', histtype="step",

         linewidth=5, label='Cohort 1', density='True')

plt.hist(df2.age, color='white', edgecolor='blue', histtype="step",

         linewidth=5, label='Cohort 2', density='True')

plt.hist(df3.age, color='white', edgecolor='red', histtype="step",

         linewidth=3, label='Cohort 3', density='True', linestyle=('dashed'))

plt.hist(df4.age, color='white', edgecolor='green', histtype="step",

         linewidth=3, label='Cohort 4', density='True', linestyle=('dashed'))



legend_elements = [Line2D([0], [0], lw=5, color='hotpink', label='Cohort 1'),

                   Line2D([0], [0], lw=5, color='blue',    label='Cohort 2'),

                   Line2D([0], [0], linestyle='--', lw=5, color='red', label='Cohort 3'),

                   Line2D([0], [0], linestyle='--', lw=5, color='green', label='Cohort 4'),

                  ]





plt.legend(handles=legend_elements, loc='upper left')

plt.xlabel('Age')

plt.title("Normalized Age Distribution")



# Compare normalized BMI distribution of the 4 cohorts

fig, ax = plt.subplots(figsize=(10,5))

plt.hist(df1.BMI, color='white', edgecolor='hotpink', histtype="step",

         linewidth=5, label='Cohort 1', density='True')

plt.hist(df2.BMI, color='white', edgecolor='blue', histtype="step",

         linewidth=5, label='Cohort 2', density='True')

plt.hist(df3.BMI, color='white', edgecolor='red', histtype="step",

         linewidth=3,  label='Cohort 3', density='True', linestyle=('dashed'))

plt.hist(df4.BMI, color='white', edgecolor='green', histtype="step",

         linewidth=3, label='Cohort 4', density='True', linestyle=('dashed'))

plt.legend(handles=legend_elements, loc='upper left')

plt.xlabel('BMI')

plt.title("Normalized BMI Distribution")
###############################################################

#Study which OTU are significantly related to cancer in patients

# Use PCA to reduce narrow the set of impactful OTU

#############################################################

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



def perform_pca(df):

    Y = df["diagnosis_int"]

    # Drop clinical phenotype information 

    df_OTU=df.drop(["diagnosis_int", "Sample.ID","gender_int", "age", "BMI", "obese"], axis=1)

    # Standardise the OTU columns so that we compare their variance at the same scale. 

    OTU=StandardScaler().fit_transform(df_OTU)

    # Convert standardised data from numpy to dataframe

    df_OTU_scaled=pd.DataFrame(OTU)

    #df_bac_scaled.head(3)

    

    # Keep components that explain 70% variance of the data

    # Chose 70% because the variance plot started flattening at higher %

    pca = PCA(0.6)

    Bac_scaled_pca = pca.fit_transform(df_OTU_scaled)

    var_ratio=pca.explained_variance_ratio_

    firstPCA=pca.components_[0]

    return Bac_scaled_pca, var_ratio, Y, firstPCA



Bac1_scaled_pca, var_ratio1, Y1, first_pca1= perform_pca(df1)



# Find the explained variance ratio of the components

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(np.arange(1,10), var_ratio1, color = 'coral')

ticks = np.arange(1,10)

ax.set_xticks(ticks)

ax.set_xlabel('Principle components')

ax.set_ylabel('Explained variance')



# plotting first 4 PCAs against each other for 1st cohort

fig, ax = plt.subplots(3,3, figsize=(15,15))

fig.suptitle("PCAs against each other", fontsize=25)

for i in range(3):

    for j in range(3):

        ax[i,j].scatter(Bac1_scaled_pca[:, i], Bac1_scaled_pca[:, j], c=Y1, cmap='RdYlGn')

        ax[i,j].set_xlim(-Bac1_scaled_pca[:, i].std(), Bac1_scaled_pca[:, i].std())

        ax[i,j].set_ylim(-Bac1_scaled_pca[:, j].std(), Bac1_scaled_pca[:, j].std())

ax[0,0].set_ylabel('PCA 1')

ax[1,0].set_ylabel('PCA 2')

ax[2,0].set_ylabel('PCA 3')

ax[2,0].set_xlabel('PCA 1')

ax[2,1].set_xlabel('PCA 2')

ax[2,2].set_xlabel('PCA 3')







#try t-SNE approach



from sklearn.manifold import TSNE



if False:

 #Y = df1["diagnosis_int"]

 #df1_OTU=df1.drop(["diagnosis_int", "Sample.ID","gender_int", "age", "BMI", "obese"], axis=1)

 #OTU1=StandardScaler().fit_transform(df1_OTU)

 #df_OTU1_scaled=pd.DataFrame(OTU1)

 #tsnes = TSNE(random_state=42).fit_transform(df_OTU1_scaled) #default output 2 components

 #plt.scatter(tsnes[:,0], tsnes[:,1], c=Y, cmap='RdYlGn')



 Bac1_scaled_pca, var_ratio1, Y1, first_pca1= perform_pca(df1)

 tsnes = TSNE(random_state=42).fit_transform(Bac1_scaled_pca)

 plt.scatter(tsnes[:,0], tsnes[:,1], c=Y, cmap='RdYlGn')

# Compare first PCAs of the different cohorts 

Bac2_scaled_pca, var_ratio2, Y2, first_pca2 = perform_pca(df2)

Bac3_scaled_pca, var_ratio3, Y3, first_pca3 = perform_pca(df3)

Bac4_scaled_pca, var_ratio4, Y4, first_pca4 = perform_pca(df4)



# plot distributon of weights of OTUs

fig, ax = plt.subplots(figsize=(10,5))

plt.hist(first_pca1, color='white', edgecolor='hotpink', histtype="step", linewidth=6, label='Cohort 1')

plt.hist(first_pca2, color='white', edgecolor='blue', histtype="step", linewidth=6, label='Cohort 2')

plt.hist(first_pca3, color='white', edgecolor='red', histtype="step", linestyle=('dashed'), linewidth=6, label='Cohort 3')

plt.hist(first_pca4, color='white', edgecolor='green', histtype="step", linestyle=('dashed'), linewidth=6, label='Cohort 4')

plt.legend(handles=legend_elements, loc='upper left')

plt.xlabel('Weights of OTUs')

plt.title("Histogram of weights components of first PCA")



# Apply a loose cut of weight > 0.03 to select the main OTUs which explain the variance

fig, ax = plt.subplots(figsize=(10,5))

# First plot the original

plt.plot(first_pca1, alpha=0.5)

plt.xlabel('OTUs')

plt.ylabel('Weights')



#Plot the filtered array

first_pca1_filtered=first_pca1.copy()

first_pca1_filtered[first_pca1_filtered<0.034]=np.nan

plt.plot(first_pca1_filtered, '.', markersize=15, markeredgecolor='w', label='Top few OTUs')

plt.legend()



top_num=100

top_OTUs_first_pca_1 = np.argsort(first_pca1)[-top_num:]

top_OTUs_first_pca_2 = np.argsort(first_pca2)[-top_num:]

top_OTUs_first_pca_3 = np.argsort(first_pca3)[-top_num:]

top_OTUs_first_pca_4 = np.argsort(first_pca4)[-top_num:]



#find common OTUs among cohorts

top_list = [top_OTUs_first_pca_1, top_OTUs_first_pca_2, 

            top_OTUs_first_pca_3, top_OTUs_first_pca_4]

similarity_matrix=np.zeros((4,4))



for i in range(4):

    for j in range(4):

        similarity_matrix[i,j] = len(top_list[i][np.in1d(top_list[i], top_list[j])])/top_num*100.

        #if i!=j:

            

            #print("common OTUs between", i, "and", j, ": ", top_list[i][np.in1d(top_list[i], top_list[j])])

print(similarity_matrix)



#Find the common OTUs in first PCA of cohort 1 and 2, as they have the 

commonTopOTU_1and2 = top_list[0][np.in1d(top_list[0], top_list[1])]

#print(commonTopOTU_1and2)
# Find the names of the common OTUs between cohort 1 and 2 with weights among top 100 for first principal component



# Get the code numbers of the OTU species from taxonomy datafile

df_taxonomy['code_num'] = df_taxonomy['OTUcode'].str.split(pat='u').str[1]

#print(df_taxonomy['code_num'].values)



# Convert code_nums to integers

temp = (df_taxonomy['code_num'].values).astype(np.int)

# Find which OTU numbers are in commonTopOTU_1and2. Print the corresponding species

print('OUT indexes, species that are common in top 100 contributors to 1st PCs of coh 1 & coh 2:')

print('\n')

for i in range(len(temp)):

    if temp[i] in commonTopOTU_1and2:

      print(df_taxonomy['OTUcode'][i], '     ', df_taxonomy['Species'][i])

#Meta-analysis



# Join Cohort 1 and 2 (no clinical information), find PCA 

frames=[df_coh1, df_coh2]

df12_joined=pd.concat(frames,join='inner', ignore_index=True)

df12_encoded = clean_data(df12_joined)

Bac12_scaled_pca, var_ratio12, Y12, first_pca12 = perform_pca(df12_encoded)



#Find top 25 OTUs in the joined dataset

top_num=25

top_OTUs_first_pca_12 = np.argsort(first_pca12)[-top_num:]



#Get metagenomic signature for joined dataset of coh1 and 2 using only bacterial information



# Get the code numbers of the OTU species from taxonomy datafile

df_taxonomy['code_num'] = df_taxonomy['OTUcode'].str.split(pat='u').str[1]

#print(df_taxonomy['code_num'].values)



# Convert code_nums to integers

temp = (df_taxonomy['code_num'].values).astype(np.int)



# Find which OTU numbers are in top_OTUs_first_pca_12. Print the corresponding species

print('OUT indexes, species that are top 25 contributors to 1st PC of metadata(coh 1+coh 2):')

print('\n')

for i in range(len(temp)):

    if temp[i] in top_OTUs_first_pca_12:

      print(df_taxonomy['OTUcode'][i], '     ', df_taxonomy['Species'][i])
#try t-SNE approach

from sklearn.manifold import TSNE



if False:

 # Join Cohort 1 and 2 (no clinical information), find PCA 

 frames=[df_coh1, df_coh2]

 df12_joined=pd.concat(frames,join='inner', ignore_index=True)

 df12_encoded = clean_data(df12_joined)

 Bac12_scaled_pca, var_ratio12, Y12, first_pca12 = perform_pca(df12_encoded)



 tsnes = TSNE(random_state=42, n_components=3).fit_transform(Bac12_scaled_pca)

 plt.scatter(tsnes[:,1], tsnes[:,2], c=Y12, cmap='RdYlGn')
#################################################

# Meta analysis with only bacterial information

#################################################

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC



RANDOM_SEED=42 # for deterministic behavior



#############################

#Declare various classifiers

##############################



forest_clf = RandomForestClassifier(random_state=RANDOM_SEED)

log_clf = LogisticRegression(random_state=RANDOM_SEED)

svm_clf = SVC(kernel="poly", gamma='auto', degree=1, C = 10, probability=True, random_state=RANDOM_SEED)

adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED)



def plot_roc_curve(fpr, tpr, ax, label=None):

    ax.plot(fpr, tpr, linewidth=4, label=label) 

    ax.set_xlim([0,1])

    ax.set_ylim([0,1])

    ax.set_xlabel('False +ve Rate')

    ax.set_ylabel('True +ve Rate')

    

# First PCA's component wt. distribution shows that 

#cohort1 & 3 (cohort 2 & 4) are similar. 



# Concatenate dataframes on common columns

# frames=[df_coh1, df_con2, df_con3, df_con4]

# frames=[df_coh1, df_coh3]



for k in range(2):

    if k==0: 

        frames=[df_coh2, df_coh4]

    else:

        frames=[df_coh1, df_coh2]

    

    df_joined=pd.concat(frames,join='inner', ignore_index=True)

    df_encoded = clean_data(df_joined)

    df_encoded.shape



    #####################################################

    # Split into training and test sets

    #####################################################

    train_set, test_set = train_test_split(df_encoded, test_size=0.2, random_state=42) 

    #train_set.describe()



    #Separate label from features

    x_train = train_set.drop(["diagnosis_int"], axis=1)

    y_train = train_set["diagnosis_int"]

    x_test = test_set.drop(["diagnosis_int"], axis=1)

    y_test = test_set["diagnosis_int"]



    # Drop clinical phenotype information

    x_train_bac=x_train.drop(["Sample.ID","gender_int", "age", "BMI", "obese"], axis=1)

    x_test_bac=x_test.drop(["Sample.ID","gender_int", "age", "BMI", "obese"], axis=1)



    # Perform standardization transformation 

    scaler = StandardScaler()

    scaler.fit(x_train_bac)

    x_trainbac_scaled = scaler.transform(x_train_bac)

    x_testbac_scaled = scaler.transform(x_test_bac)



    # PCA-transform  

    pca = PCA(0.9)

    pca.fit(x_trainbac_scaled)

    x_trainbac_new=pca.transform(x_trainbac_scaled)

    var_ratio_train=pca.explained_variance_ratio_

    

    x_testbac_new=pca.transform(x_testbac_scaled)

    var_ratio_test=pca.explained_variance_ratio_



    print("# Principal components that explain 90% variance in the data: ", x_trainbac_new.shape[1])

    #---------------------------------------------------------------------------------------

    # Evaluate PCA results on the training set

    #--------------------------------------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(10,5))

    ax1.bar(range(1,len(var_ratio_train)+1), var_ratio_train, color = 'coral')

    ticks = np.arange(1,len(var_ratio_train)+1)[::5]

    ax1.set_xticks(ticks)

    ax1.set_xlabel('Principle components')

    ax1.set_ylabel('Explained variance')



    # plotting first 4 PCAs against each other 

    if k==0:

        fig2, ax2 = plt.subplots(3,3, figsize=(10,10))

        fig2.suptitle("PCAs against each other", fontsize=25)

        for i in range(3):

            for j in range(3):

                ax2[i,j].scatter(x_trainbac_new[:, i], x_trainbac_new[:, j], c=y_train, cmap='RdYlGn')

                ax2[i,j].set_xlim(-x_trainbac_new[:, i].std(), x_trainbac_new[:, i].std())

                ax2[i,j].set_ylim(-x_trainbac_new[:, j].std(), x_trainbac_new[:, j].std())

        ax2[0,0].set_ylabel('PCA 1')

        ax2[1,0].set_ylabel('PCA 2')

        ax2[2,0].set_ylabel('PCA 3')

        ax2[2,0].set_xlabel('PCA 1')

        ax2[2,1].set_xlabel('PCA 2')

        ax2[2,2].set_xlabel('PCA 3')

    ###########################################################

    # Evaluate the performance of the classifiers

    ###########################################################

    fig3, ax3 = plt.subplots(figsize=(8,8))

    for clf, label in zip([forest_clf, log_clf, svm_clf, adaboost_clf], 

                          ['Forest', 

                          'Logistic', 

                          'Linear SVM',

                          'ADAboost']):

        # Convert probability to score and get params for ROC 

        y_score = cross_val_predict(clf, x_trainbac_new, y_train, cv=3, method="predict_proba")[:,1]

        fpr, tpr, thresholds = roc_curve(y_train, y_score)

        plot_roc_curve(fpr, tpr, ax3, label)

        if k==0:

            print("AUC (cohort 2 + 4): %0.2f [%s]" % (roc_auc_score(y_train, y_score), label))

        else:

            print("AUC (cohort 1 + 2): %0.2f [%s]" % (roc_auc_score(y_train, y_score), label))

    

    print("\n")

    plt.plot([0,1], [0,1], 'k--', label='Random Classifier')

    plt.legend()

    if k==0:

        plt.title("ROC curves, cohort 2 & 4 combined")

    else:

        plt.title("ROC curves, cohort 1 & 2 combined")
#from mlxtend.classifier import StackingCVClassifier

#lr = LogisticRegression() # for stacked classifiers

#sclf = StackingCVClassifier(classifiers=[forest_clf, log_clf, svm_clf, adaboost_clf], meta_classifier=lr)  

#scores = cross_val_score(sclf, x_trainbac_new, y_train, cv=3, scoring='accuracy')

#print("Accuracy: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), 'Stacked Classifier'))
######################################

# Apply best classifier on test sample

#######################################
################################################

# Meta analysis with bacterial + clinical info

################################################



# Add clinical information to a new dataframe

age_col = x_train.age

bmi_col = x_train.BMI

gender_col = x_train.gender_int

array_clin = np.c_[age_col, bmi_col, gender_col]

df_clin = pd.DataFrame(array_clin)

df_clin.head(5)



# Standardize clinical information

scaler = StandardScaler()

scaler.fit(df_clin)

x_clin_scaled = scaler.transform(df_clin)



# Combine clinical information with standardized principal components of OTU

x_trainfull_new = np.c_[x_trainbac_new, x_clin_scaled]



###########################################################

# Evaluate the performance of the classifiers

###########################################################

fig4, ax4 = plt.subplots(figsize=(8,8))

for clf, label in zip([forest_clf, log_clf, svm_clf, adaboost_clf], 

                      ['Forest', 

                       'Logistic', 

                       'Linear SVM',

                       'ADAboost']):

    # Convert probability to score and get params for ROC 

    y_score = cross_val_predict(clf, x_trainfull_new, y_train, cv=3, method="predict_proba")[:,1]

    fpr, tpr, thresholds = roc_curve(y_train, y_score)

    plot_roc_curve(fpr, tpr, ax4, label)

    print("AUC: %0.2f [%s]" % (roc_auc_score(y_train, y_score), label))

plt.plot([0,1], [0,1], 'k--', label='Random Classifier')

plt.legend()

plt.title("ROC curves, bacterial & clinical information combined")