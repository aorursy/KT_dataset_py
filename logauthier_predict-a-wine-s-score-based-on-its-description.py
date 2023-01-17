#Easy data manipulation

import pandas as pd

import numpy as np



#Plotting

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

import matplotlib.pyplot as plt



#Who likes warnings anyway?

import warnings

warnings.filterwarnings('ignore')



#Pre-processing, tuning of parameters and scoring tools

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve



#Basic text mining tools

from sklearn.feature_extraction import stop_words

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction import text #Allow stop_words customization



#Machine Learning models

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB



#Used for distribution fitting, and representation

from scipy import stats



#Time measuring for model training

from time import time
df = pd.read_csv('../input/winemag-data-130k-v2.csv')
print("Number of entries (rows):", df.shape[0],\

      "\nNumber of features (columns):", df.shape[1])



df.head(3)
#Keep only the useful columns and rename them for ease of use

df = df[['description', 'points']]

df.rename(columns={'description':'Description',

                   'points':'Score'},

                   inplace=True)

 

#Add a column with the length of the description in characters, we use it to check for empty descriptions

df["Description_Length"] = [len(desc) for desc in df['Description']]



#Check for missing values

print("Number of missing values for the Score feature: ", len(df[df['Score'].isnull()]))

print("Number of missing descriptions: ", len(df[df['Description_Length']==0]))
df.describe()
#Make a subplot grid

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)})

 

#Associate a plot to each of the subplot grid

sns.boxplot(df["Score"], ax=ax_box).set_title("Basic representation of the Score feature\n")

sns.distplot(df["Score"], ax=ax_hist, kde=False, fit=stats.gamma, bins=20) 

#We can fit a gamma distribution, just for the sake of representation.

 

#Set axes legends

ax_box.set(xlabel='') #Remove x axis name for the boxplot

ax_hist.set(ylabel='Density')



plt.show()

Q3 = np.quantile(df['Score'], 0.75) #Third quartile

Q1 = np.quantile(df['Score'], 0.25) #First quartile

IQR = Q3 - Q1 #Inter Quartile Range



outlier_score_threshold =  Q3 + 1.5 * IQR

outlier_number=len(df[ df['Score'] > outlier_score_threshold ])



print("Number of outliers:", outlier_number,

      "\nOutlier proportion:", round(outlier_number/len(df['Score'])*100, 3),"%",

      "\nOutlier threshold score:", outlier_score_threshold,"/ 100")
#Make a subplot grid

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)})

 

#Associate a plot to each of the subplot grid

sns.boxplot(df["Description_Length"], ax=ax_box).set_title("Basic description of the Description_Length")

sns.distplot(df["Description_Length"], ax=ax_hist, kde=False, fit=stats.gamma, bins=50) 

#We can fit a gamma distribution, the fit is rather impressive here.



#Parameters of the gamma distribution

alpha, loc, beta = stats.gamma.fit(df['Description_Length'])

plt.legend(['Gamma distribution \nShape = {0:.2f} \nLoc = {1:.2f}  \nScale = {2:.2f}'.format(alpha, loc, beta)],loc='best')



#Set axes legends

ax_box.set(xlabel='') #Remove x axis name for the boxplot

ax_hist.set(ylabel='Frequency')



plt.show()



#How good is the fit with the gamma distribution ?

fig = plt.figure()

res = stats.probplot(df['Description_Length'], dist=stats.gamma(a= alpha, loc=loc, scale=beta), plot=plt)

plt.show()



#Other basic statistics

print("Skewness: %f" % df['Description_Length'].skew())

print("Kurtosis: %f" % df['Description_Length'].kurt())
Q3 = np.quantile(df['Description_Length'], 0.75) #Third quartile

Q1 = np.quantile(df['Description_Length'], 0.25) #First quartile

IQR = Q3 - Q1 #Inter Quartile Range



outlier_score_threshold_high = Q3 + 1.5 * IQR

outlier_score_threshold_low = Q1 - 1.5 * IQR



outlier_number_total=len(df[np.logical_or(df['Description_Length'] > outlier_score_threshold_high,

                         df['Description_Length'] < outlier_score_threshold_low)])



outlier_number_low = len(df[df['Description_Length'] < outlier_score_threshold_low])

outlier_number_high = outlier_number_total - outlier_number_low



print("Number of outliers (high - low):", outlier_number_total, "(",outlier_number_high,"-",outlier_number_low,")",

      "\nOutlier proportion:", round(outlier_number_total/len(df['Description_Length'])*100, 3),"%",

      "\nOutlier threshold lengths (high - low):", outlier_score_threshold_high,"-",outlier_score_threshold_low)
#These are the entries with a very short description

sub_df = df[df['Description_Length']<outlier_number_low]



#Make a subplot grid

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)})

 

#Associate a plot to each of the subplot grid

sns.boxplot(sub_df['Score'], ax=ax_box).set_title("Scores of wines given a description shorter than 70 words\n")

sns.distplot(sub_df['Score'], ax=ax_hist, kde=False, fit=stats.gamma, bins=15) 

 

#Set axes legends

ax_box.set(xlabel='') #Remove x axis name for the boxplot

ax_hist.set(ylabel='Frequency')



plt.show()



mean_score = np.mean(sub_df['Score'])

print("Mean score given after a description shorter than 70 words:", round(mean_score,2), "/ 100")



proportion_below_median = len(sub_df[sub_df['Score']<88]['Score'])/len(sub_df['Score'])

print("Proportion of wines with short description with a score below the set's median:", round(proportion_below_median*100,2),"%")

sns.jointplot(x="Score", y="Description_Length", data=df)

plt.show()



corr= np.corrcoef(df["Score"], df["Description_Length"])[0,1]

print("Correlation between Score and Description_Length:",round(corr,2))
corpus = df["Description"].values

Y = df["Score"].values
#Customize stop words after having a first look at the most frequent words

customStopWords = text.ENGLISH_STOP_WORDS.union(['wine', '2009', '2010','2011', '2012', '2013', '2014', '2015','2016', '2017', '2018',

                                                 '2019', '2020', '2021', '2022','2023', '2024', '2025', '2030', '100', '10', '12',

                                                 '14', '15', '20', '25', '30','40', '50', '60', '70', '90'])

#The words we add to the english stop words are mostly references to dates and prices, hence numbers.



#Use the CountVectorizer: we consider 1000 features, either individual words or pairs

CV = CountVectorizer(stop_words=customStopWords, max_features=1000, ngram_range=(1,2))

X = CV.fit_transform(corpus) #Let's be careful here, X is a sparse Matrix



print("Number of entries (rows):", X.shape[0],\

      "\nNumber of features (columns):", X.shape[1])
X_array = X.toarray() #Convert X from a sparse matrix to a usual matrix



inverted_dict = dict([[v,k] for k,v in CV.vocabulary_.items()]) # {X_array column index: "word"}

final_dict = {} # {"word": total number of instances }



for x in range(len(X_array[0,:])): #Fill the final dict

    final_dict[inverted_dict[x]]=np.sum(X_array[:,x]) 



print("20 most frequent words:",sorted(final_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[0:20]) #Display of the final dict
TFIDF = TfidfTransformer()

X_TFIDF = TFIDF.fit_transform(X)
Y_rep = [0] #Will stock the cumulative explained variance ratio of the SVD, for each n_components



#Actual SVD

SVD = TruncatedSVD(n_components = 999) 

X_SVD = SVD.fit_transform(X_TFIDF)

var = SVD.explained_variance_ratio_



#This will help us decide on the final number of components we want to keep 

for x in range(999): 

    Y_rep.append(Y_rep[-1]+var[x])



plt.plot(Y_rep)

plt.plot(sorted(var*78, reverse=True))



plt.title("Explained variance ratio, and scaled marginal explained variance gain\nw.r.t the number of components kept for SVD\n")

plt.legend(['Explained variance ratio', 'Marginal gain of explained variance'], loc='best')

plt.xlabel('Number of components')

plt.ylabel('')



plt.show() 
#Balanced discretization in 2 classes

median = np.median(Y) #88.0

Y[Y < median] = 0

Y[Y >= median] = 1



#Re-type Y as int

Y=Y.astype(int)



#Concatenation of X_TFIDF and Description_Length to obtain our final X matrix

X = np.append(X_TFIDF.toarray(), df["Description_Length"].values[:, None], axis=1)



print("Number of entries:", X.shape[0],\

      "\nNumber of features:", X.shape[1])
#Test/Train split

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3, shuffle=True)
#Normalization

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)    #We don't cheat here, we fit our scaler only on the train data

X_test = scaler.transform(X_test)          #X_test is ONLY transformed, based on X_train fit
def FitAndScore(classifier, X_train, X_test, y_train, y_test):

    print("Fitting...")

    start = time()

    classifier.fit(X_train, y_train)

    end = time()

    print("( {0:.2f} seconds )".format(end-start))

    

    print("\nPredicting...")

    start = time()

    y_predicted = classifier.predict(X_test)

    end = time()

    print("( {0:.2f} seconds )".format(end-start))

    

    print("\nReporting...\n")

    print(classification_report(y_test, y_predicted),"\n")

    print("Confusion matrix:\n")

    print(confusion_matrix(y_test, y_predicted),"\n")

    print("Cohen's Kappa score : ",cohen_kappa_score(y_test, y_predicted),"\n")

    

    #If we formulate the problem with binary classes, we can take a look at the ROC curve and associated score

    if len(np.unique(y_test)) == 2:

        print("AUC score : {0:.3f}".format(roc_auc_score(y_test, y_predicted)))

        fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

        plt.plot([0, 1], [0, 1], linestyle='--')

        plt.plot(fpr, tpr, marker='.')

        plt.title("ROC Curve")

        plt.show()
dt = DecisionTreeClassifier(criterion='gini', max_depth=10)

FitAndScore(dt, X_train, X_test, y_train, y_test)
#We have negative values, so let's scale everyting between 0 and 1 to use naive bayes classifier

mmscaler = MinMaxScaler()

X_train_01 = mmscaler.fit_transform(X_train)    #Still no cheating with this scaling

X_test_01 = mmscaler.transform(X_test)          



nb = MultinomialNB()

FitAndScore(nb, X_train_01, X_test_01, y_train, y_test)
LSVC = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=False,

                 intercept_scaling=1, loss='squared_hinge', max_iter=1000,

                 multi_class='ovr', penalty='l2', tol=1e-05, verbose=2)



FitAndScore(LSVC, X_train, X_test, y_train, y_test)
rf = RandomForestClassifier(n_estimators = 100, bootstrap=True, n_jobs=-1)

FitAndScore(rf, X_train, X_test, y_train, y_test)
features = CV.vocabulary_

features['Description_Length']=1000



importances = rf.feature_importances_



features_inv = dict([[v,k] for k,v in features.items()]) #Structure {X index:'feature name'}

final_features_dict = {} #Structure {'feature name':relative importance}

for i in range(len(importances)):

    final_features_dict[features_inv[i]] = importances[i]



sorted_feature_importance = sorted(final_features_dict.items(), key = lambda kv:[kv[1], kv[0]], reverse=True)

sorted_feature_importance[0:20]
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1);

bclf = AdaBoostClassifier(base_estimator=clf, n_estimators=10)

FitAndScore(bclf, X_train, X_test, y_train, y_test)
mlp = MLPClassifier(activation='logistic', alpha=1e-03, batch_size='auto',\

                    beta_1=0.9, beta_2=0.999, early_stopping=False,\

                    epsilon=1e-08, hidden_layer_sizes=(600,400),\

                    learning_rate='constant', learning_rate_init=0.0001,\

                    max_iter=200, momentum=0.9, n_iter_no_change=10,\

                    nesterovs_momentum=True, power_t=0.5,\

                    shuffle=True, solver='adam', tol=0.00001,\

                    validation_fraction=0.1, verbose=True, warm_start=False)



FitAndScore(mlp, X_train, X_test, y_train, y_test)