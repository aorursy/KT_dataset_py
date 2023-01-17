import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split

verbose = False

#alpha is a parameter to control how much of the final dataset is used in training the random forest classifier
alpha = 0.9

#n_models determines how many times to build and test a classifier from the final dataset
n_models = 20

#this is a random parameter to control the random state of the model, for reproducibility.
rnd=0
def relative_weights(df, class_name, col_name, delimiter, synonyms ={}, reduce = False):
    #df is the input dataframe
    #col_name is the name of the column whose relative weights we want to compute
    #delimiter is how the entries in col_name are divided into classes
    #synonyms lists any synonyms among words that may appear in col_name entries, e.g. not provided = not specified

    counts = {}
    for index, row in df.iterrows():
        #extract the words for this row
        new_words = row[col_name].split(delimiter)
        #make synonym substitutions and remove any duplicates
        new_words = list(set([synonyms.get(j,j) for j in new_words]))
        for n in new_words:
            #for existing words, we increase the overall count and the class frequency
            #if the word is new, create an entry [1, (class of this row)]
            if n in counts:
                counts[n][0] +=1
                counts[n][1] += row[class_name]
            else:
                counts[n] = [1,row[class_name]]
    n_entries = df.shape[0]
    weights = dict([(key, (np.power((2*counts[key][1]/counts[key][0] -1),2), np.floor(0.5 + counts[key][1]/counts[key][0]))) 
                      for key in counts])

    #create a subfunction to get the weight of a given entry
    #this will allow us to use a lambda function to map over rows of the dataframe
    def get_weight(row, dictionary, delimiter_sub, syns):
        kwds = row.split(delimiter_sub)
        #make synonym substitutions
        kwds = list(set([syns.get(j,j) for j in kwds]))
        return np.sum([np.power((-1),dictionary[j][1]) *dictionary[j][0] for j in kwds])/len(kwds)
        
    return df[col_name].apply(lambda x: get_weight(x, weights, delimiter,synonyms))

#for converting Chromosome numbers to strings
def to_int(s):
    try:
        return int(s)
    except ValueError:
        if s == 'X':
            return 23
        else:
            return 24
    
#for shortening base-pair entries
def string_to_short(s):
    if len(s)>1:
        return 'L'
    else:
        return s
    
#Extract training and testing data from a dataset
def get_train_test(df, alpha = 0.9, clss=None):
    train = df.sample(frac = 0.9, axis = 0)
    test = df[~df.index.isin(train.index)]
    
    cols_not_class = [j for j in df.columns.values]
    if clss in cols_not_class:
        cols_not_class.remove(clss)
    return train[cols_not_class], train[clss], test[cols_not_class], test[clss]
data = pd.read_csv("../input/clinvar_conflicting.csv")

print("Processing data...")

#drop NA values up to 50%
data = data.dropna(axis=1, thresh=.5*data.shape[0])

#convert entries in the Chromosome column to integers
data.CHROM = pd.Series(data.CHROM.map(lambda x: to_int(x)), index = data.CHROM.index)

#drop rows with few NA features in certain columns
data = data.drop(data[ (data['STRAND'].isna()) | (data['BIOTYPE'].isna()) | (data['SYMBOL'].isna())].index, axis = 0)

#'Feature Type' has a unique entry, so we drop this feature
data = data.drop(['Feature_type'], axis = 1)

#start looking at single / multiple base entries. We see most entries are single bases
single_base = ['T','A','G','C','-']

I = data[~data.REF.isin(single_base)].index
I=I.union(data[~data.ALT.isin(single_base)].index)
I=I.union(data[~data.Allele.isin(single_base)].index)

#We'll address this in two ways: we'll add a feature that counts the length of the bases involved in the REF, ALT, Allele category
#We'll also condense the longer base sequences into a category 'L' -- this allows the distinctions between G,C,T,A,- to still appear in thse features    
data['REF_LEN'] = pd.Series(data['REF'].apply(lambda x: len(x)), index = data['REF'].index)
data['ALT_LEN'] = pd.Series(data['ALT'].apply(lambda x: len(x)), index = data['ALT'].index)
data['Allele_LEN'] = pd.Series(data['Allele'].apply(lambda x: len(x)), index = data['Allele'].index)
    
data['REF'] = pd.Series(data['REF'].apply(string_to_short), data['REF'].index)
data['ALT'] = pd.Series(data['ALT'].apply(string_to_short), data['ALT'].index)
data['Allele'] = pd.Series(data['Allele'].apply(string_to_short), data['Allele'].index)

#Here we start with some imputation
imp_mean = SimpleImputer(strategy = 'mean')
imp_med = SimpleImputer(strategy = 'median')

data['ORIGIN'] = pd.Series(imp_med.fit_transform(data.ORIGIN.values.reshape((-1,1))).flatten(), index = data.ORIGIN.index)
data['LoFtool'] = pd.Series(imp_mean.fit_transform(data.LoFtool.values.reshape((-1,1))).flatten(), index = data.LoFtool.index)
data['CADD_PHRED'] = pd.Series(imp_mean.fit_transform(data.CADD_PHRED.values.reshape((-1,1))).flatten(), index = data.CADD_PHRED.index)
data['CADD_RAW'] = pd.Series(imp_mean.fit_transform(data.CADD_RAW.values.reshape((-1,1))).flatten(), index = data.CADD_RAW.index)

#Finally, we normalize numerical columns
cols_to_normalize = ['AF_ESP', 'AF_EXAC', 'AF_TGP', 'LoFtool', 'CADD_PHRED', 'CADD_RAW']
for c in cols_to_normalize:
    data[c] = pd.Series((lambda x: (x-x.min())/(x.max() - x.min()))(data[c]), index = data[c].index)

#We also normalize the position factor
#within each chromosome, we rescale the position to a numerical value between 0 and 1 representing its relative position in the chromosome
#We'll use the largest position for a given chromosome as a proxy for its actual maximal length; this needs to be replaced with actual maximal lengths
Maxes = [data[data['CHROM']==j].POS.max() for j in data.CHROM.unique()]
data.POS = data.apply(lambda x: x.POS / Maxes[x.CHROM-1], axis = 1)
    
#replace NaN entries in the Codons, Amino_Acids, and MC columns with a class 'unkn'.
data.Codons.fillna('unkn', inplace=True)
data.Amino_acids.fillna('unkn', inplace=True)
data.MC.fillna('unkn', inplace=True)

#convert the impact scores to numerical values
impact_dict = {"MODIFIER":0, "LOW": 0.33333, "MODERATE": 0.66667, "HIGH": 1}
data["IMPACT"] = data["IMPACT"].apply(lambda x:impact_dict[x])
    
#create the weights column
data['CLNDN_WTS'] = relative_weights(data, "CLASS", "CLNDN", "|", {"not_specified" : "not_provided"})
data['MC_WTS'] = relative_weights(data, "CLASS", "MC", ",")
data['Consequence_WTS'] = relative_weights(data, "CLASS", "Consequence", "&")
    
    
print("Finished!")
#Some summary info about data so far
if verbose:
    for c in data.columns:
        print("Column: ", c, "\n")
        print(data[c].unique())
        print("\n")
    sns.countplot(x="CLASS", data = data)
pd.Series(sorted(data.CLNDN_WTS.values, reverse = True)).plot()
pd.Series(sorted(data.MC_WTS.values, reverse = True)).plot()
pd.Series(sorted(data.Consequence_WTS.values, reverse = True)).plot()
#select features for the model
features_columns = ["CHROM", "POS", "REF", "ALT", "AF_ESP", "AF_EXAC", "AF_TGP", "CLNVC", "ORIGIN", "CLASS", "Allele", "IMPACT", "STRAND",
                   "LoFtool", "CADD_PHRED", "CADD_RAW", "REF_LEN", "ALT_LEN","Allele_LEN", "CLNDN_WTS", "MC_WTS", "Consequence_WTS"]

#here we extract a balanced dataset:
data_balanced = pd.concat([data[data['CLASS']==0].sample(n=data[data['CLASS']==1].shape[0], axis = 0), data[data['CLASS']==1]])

#encode categorical data and extract train / test data
data_bal_encoded = pd.get_dummies(data_balanced[features_columns])
features = data_bal_encoded[data_bal_encoded.columns.drop("CLASS")]
labels = data_bal_encoded.CLASS

#model_parameters
a=[150, 15]
clf = RandomForestClassifier(n_estimators = a[0], min_samples_split = a[1])

plt.figure()

AUCS = []
f1s =[]

for i in range(n_models):
    #split and fit the data
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size = alpha, test_size = 1-alpha)
    roc_probs = clf.fit(features_train, labels_train).predict_proba(features_test)
    y_pred = clf.predict(features_test)
    
    #feedback from this model
    print("Model %i trained"%(i+1))
    if verbose: 
        print(classification_report(labels_test, y_pred))
        print("Model %i AUROC score: %0.3f"%((i+1), auc(fpr,tpr)))
    
    #capture data
    fpr, tpr, threshs = roc_curve(labels_test.values, roc_probs[:,1])
    plt.plot(fpr, tpr)
    AUCS.append(auc(fpr,tpr))
    
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.suptitle("AUROC Curves")
plt.title("Average AUROC: %0.3f; Standard Deviation: %0.3f"%(np.mean(AUCS), np.std(AUCS)))
plt.show()