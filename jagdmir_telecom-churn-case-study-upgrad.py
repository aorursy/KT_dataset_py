# lets import the required libraries and packages

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score

import warnings

warnings.filterwarnings('ignore')
# lets import the dataset

telecom = pd.read_csv("../input/telecom/telecom_churn_data.csv")

telecom.head()
# lets check the dimensions of the dataset

telecom.shape
# High-value customers : Those who have recharged with an amount more than or equal to X, where X is the 70th percentile of 

# the average recharge amount in the first two months (the good phase)



# lets dervie features to extract high value customers

# lets find out total amount spent by customers on data recharge,we have two colums available to find this out 

# first column is av_rech_amt_data_x (x represents month here, would be either 6 or 7 or 8)

# second column is total_rech_data_x (x represnts month here, would be either 6 or 7 or 8)

# lets introduce a new column total_rech_data_amt_x which can be calculated as av_rech_amt_data_x * total_rech_data_x



telecom['total_rech_data_amt_6'] = telecom['av_rech_amt_data_6'] * telecom['total_rech_data_6']

telecom['total_rech_data_amt_7'] = telecom['av_rech_amt_data_7'] * telecom['total_rech_data_7']

telecom['total_rech_data_amt_8'] = telecom['av_rech_amt_data_8'] * telecom['total_rech_data_8']

telecom['total_rech_data_amt_9'] = telecom['av_rech_amt_data_9'] * telecom['total_rech_data_9']



# now we dont need columns av_rech_amt_data_x,total_rech_data_x (x = 6/7/8) , lets drop them

telecom.drop(['total_rech_data_6','total_rech_data_7','total_rech_data_8','total_rech_data_9',

'av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8','av_rech_amt_data_9'],axis = 1,inplace = True)



# lets find out the average recharge done in the first two months(june & july) - the good phase

# total amount spend would be the sum of total data recharge done & total call/sms recharges

telecom_av_rech_6n7 = (telecom['total_rech_amt_6'].fillna(0) 

+ telecom['total_rech_amt_7'].fillna(0) 

+ telecom['total_rech_data_amt_6'].fillna(0) 

+ telecom['total_rech_data_amt_7'].fillna(0))/2



# take 70 percentile of the calculated average amount

percentile_70_6n7 = np.percentile(telecom_av_rech_6n7, 70.0)

print("70 percentile is : ", percentile_70_6n7)



# fitler the given data set based on 70th percentile

telecom_hv_cust = telecom[telecom_av_rech_6n7 >= percentile_70_6n7]



print("Dimensions of the filtered dataset:",telecom_hv_cust.shape)
# lets introduce a new column "churn", values would be either 1 (churn) or 0 (non-churn)

# we will calculate churn/non-churn based on the usage as mentioned in the problem statement

telecom_hv_cust['churn'] = np.where(telecom_hv_cust[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']].sum(axis=1) == 0, 1,0)

telecom_hv_cust.head()
# lets find out churn/non churn percentage

telecom_hv_cust['churn'].value_counts()/len(telecom_hv_cust)*100



#observation : 91% of the customers do not churn, this might be a case of class imbalance, we will treat it later
# lets check the columns with no variance in their values and drop such columns

for i in telecom_hv_cust.columns:

    if telecom_hv_cust[i].nunique() == 1:

        print("\nColumn",i,"has no variance and contains only", telecom_hv_cust[i].nunique(),"unique value")

        print("Dropping the column",i)

        telecom_hv_cust.drop(i,axis=1,inplace = True)



print("\nDimension of the updated dataset:",telecom_hv_cust.shape)
# lets check the null values present in the dataset

(telecom_hv_cust.isnull().sum() * 100 / len(telecom_hv_cust)).sort_values(ascending = False)
# Drop Columns with > 30% of missing values except 9th Month's columns

cols = telecom_hv_cust.columns

telecom_null_perc = telecom_hv_cust.isnull().sum() * 100 / len(telecom_hv_cust)

telecom_null_df = pd.DataFrame({'col_name': cols,

                                 'perc_null': telecom_null_perc})



drop_cols = telecom_null_df.loc[(telecom_null_df["col_name"].str.contains('_9')==False) & (telecom_null_df["perc_null"] > 30.0)]["col_name"]

print("list of columns dropped:",drop_cols)



# lets drop these columns

telecom_hv_cust.drop(drop_cols, axis=1,inplace = True)

telecom_hv_cust.shape
# lets check for columns that can be changed to integers, floats or date types

object_col_data = telecom_hv_cust.select_dtypes(include=['object'])

print(object_col_data.iloc[0])



# observation : all the columns below can be converted to date type
# convert to datetime

for col in object_col_data.columns:

    telecom_hv_cust[col] = pd.to_datetime(telecom_hv_cust[col])



telecom_hv_cust.shape
# lets check the correlation amongst the features, drop the highly correlated ones

cor = telecom_hv_cust.corr()

cor.loc[:,:] = np.tril(cor, k=-1)

cor = cor.stack()

cor[(cor > 0.60) | (cor < -0.60)].sort_values()
# we will drop the columns with high correlation (+/- 60%)

drop_col_list = ['loc_og_t2m_mou_6','std_og_t2t_mou_6','std_og_t2t_mou_7','std_og_t2t_mou_8','std_og_t2t_mou_9','std_og_t2m_mou_6',

                'std_og_t2m_mou_7','std_og_t2m_mou_8','std_og_t2m_mou_9','total_og_mou_6','total_og_mou_7','total_og_mou_8',

                'loc_ic_t2t_mou_6','loc_ic_t2t_mou_7','loc_ic_t2t_mou_8','loc_ic_t2t_mou_9','loc_ic_t2m_mou_6','loc_ic_t2m_mou_7','loc_ic_t2m_mou_8','loc_ic_t2m_mou_9',

                'std_ic_t2m_mou_6','std_ic_t2m_mou_7','std_ic_t2m_mou_8','std_ic_t2m_mou_9','total_ic_mou_6','total_ic_mou_7','total_ic_mou_8',

                'total_rech_amt_6','total_rech_amt_7','total_rech_amt_8','total_rech_amt_9','arpu_2g_9','count_rech_2g_9','count_rech_3g_9','vol_3g_mb_6','vol_3g_mb_7','vol_3g_mb_8',

                'loc_og_t2t_mou_6','loc_og_t2t_mou_7','loc_og_t2t_mou_8','loc_og_t2t_mou_9','loc_og_t2f_mou_6','loc_og_t2f_mou_7','loc_og_t2f_mou_8','loc_og_t2f_mou_9',

                'loc_og_t2m_mou_6','loc_og_t2m_mou_7','loc_og_t2m_mou_8','loc_og_t2m_mou_9','loc_ic_t2f_mou_6','loc_ic_t2f_mou_7','loc_ic_t2f_mou_8','loc_ic_t2f_mou_9',

                'date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8']

                 

telecom_hv_cust.drop(drop_col_list, axis=1, inplace=True)

telecom_hv_cust.shape
# Now we will delete 9th month columns because we would predict churn/non-churn later based on data from the 1st 3 months

cols_to_drop = [col for col in telecom_hv_cust.columns if '_9' in col]

print(cols_to_drop)



telecom_hv_cust.drop(cols_to_drop, axis=1, inplace=True)



telecom_hv_cust.shape
# lets check the dataset again

(telecom_hv_cust.isnull().sum() * 100 / len(telecom_hv_cust)).sort_values(ascending = False)



# Obervation : we are left with few columns with around 4% of null values
# drop rows with null values

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['onnet_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['onnet_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['onnet_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['offnet_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['offnet_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['offnet_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['roam_ic_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['roam_ic_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['roam_ic_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['roam_og_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['roam_og_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['roam_og_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_og_t2c_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_og_t2c_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_og_t2c_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_og_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_og_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_og_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_og_t2f_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_og_t2f_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_og_t2f_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_og_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_og_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_og_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['isd_og_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['isd_og_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['isd_og_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['spl_og_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['spl_og_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['spl_og_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['og_others_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['og_others_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['og_others_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_ic_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_ic_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['loc_ic_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_t2t_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_t2t_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_t2t_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_t2f_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_t2f_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_t2f_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['std_ic_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['spl_ic_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['spl_ic_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['spl_ic_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['isd_ic_mou_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['isd_ic_mou_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['isd_ic_mou_8'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['ic_others_6'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['ic_others_7'])]

telecom_hv_cust = telecom_hv_cust[~np.isnan(telecom_hv_cust['ic_others_8'])]
# lets check the dataset again

(telecom_hv_cust.isnull().sum() * 100 / len(telecom_hv_cust)).sort_values(ascending = False)
# lets write a function to plot historgram for some sample columns

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# function to plot correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = "Telecom Churn"

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# function to plot scatter plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

# call the function to plot the graphs

plotPerColumnDistribution(telecom_hv_cust, 10, 5)
plotCorrelationMatrix(telecom_hv_cust, 53)
plotScatterMatrix(telecom_hv_cust, 20, 10)
# create a new colulmn, which would be average  of 6th & 7th months

# lets first create list of columns belonging to 6th and 7th months

col_list = telecom_hv_cust.filter(regex='_6|_7').columns.str[:-2]

col_list.unique()



print (telecom_hv_cust.shape)



# lets take the average now

for idx, col in enumerate(col_list.unique()):

    avg_col_name = "avg_"+col+"_av67" # lets create the column name dynamically

    col_6 = col+"_6"

    col_7 = col+"_7"

    telecom_hv_cust[avg_col_name] = (telecom_hv_cust[col_6]  + telecom_hv_cust[col_7])/ 2



# we dont need columns from which we have derived new features, we will drop those columns

print ("dimension of the updated dataset after creating dervied features:",telecom_hv_cust.shape)

col_to_drop = telecom_hv_cust.filter(regex='_6|_7').columns

telecom_hv_cust.drop(col_to_drop, axis=1, inplace=True)



print("dimension of the dataset after dropping un-necessary columns:",telecom_hv_cust.shape)
# lets now conevrt AON in months

telecom_hv_cust['aon_mon'] = telecom_hv_cust['aon']/30

telecom_hv_cust.drop('aon', axis=1, inplace=True)

telecom_hv_cust['aon_mon'].head()
# lets again draw the plots with the updated dataset

plotPerColumnDistribution(telecom_hv_cust, 10, 5)
plotCorrelationMatrix(telecom_hv_cust, 53)
plotScatterMatrix(telecom_hv_cust, 20, 10)
ax = sns.distplot(telecom_hv_cust['aon_mon'], hist=True, kde=False, 

             bins=int(180/5), color = 'purple', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 10})

ax.set_ylabel('No of Customers')

ax.set_xlabel('Tenure in months')

ax.set_title('Tenure Graph')

# below graph simply shows the tenure of the customers
tn_range = [0, 6, 12, 24, 60, 61]

tn_label = [ '0-6 Months', '6-12 Months', '1-2 Yrs', '2-5 Yrs', '5 Yrs and above']

telecom_hv_cust['tenure_range'] = pd.cut(telecom_hv_cust['aon_mon'], tn_range, labels=tn_label)

telecom_hv_cust['tenure_range'].head()
# lets check correlation of churn with other columns

plt.figure(figsize=(20,10))

telecom_hv_cust.corr()['churn'].sort_values(ascending = False).plot(kind='bar')



# observations : 

# 1. Avg Outgoing Calls & calls on romaning for 6 & 7th months are positively correlated with churn. 

# 2. Avg Revenue, No. Of Recharge for 8th month has negative correlation with churn.
# lets now draw a scatter plot between total recharge and avg revenue for the 8th month

telecom_hv_cust[['total_rech_num_8', 'arpu_8']].plot.scatter(x = 'total_rech_num_8',

                                                              y='arpu_8')
# plot between tenure and revenue

telecom_hv_cust[['aon_mon', 'avg_arpu_av67']].plot.scatter(x = 'aon_mon',

                                                              y='avg_arpu_av67')
sns.boxplot(x = telecom_hv_cust.churn, y = telecom_hv_cust.aon_mon)



# from the below plot , its clear tenured customers do no churn and they keep availing telecom services
# churn Vs Base Cost

ax = sns.kdeplot(telecom_hv_cust.avg_max_rech_amt_av67[(telecom_hv_cust["churn"] == 0)],

                color="Red", shade = True)

ax = sns.kdeplot(telecom_hv_cust.avg_max_rech_amt_av67[(telecom_hv_cust["churn"] == 1)],

                ax =ax, color="Green", shade= True)

ax.legend(["No-Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('Volume based cost')

ax.set_title('Churn Vs Base Cost')
# churn vs max rechare amount

ax = sns.kdeplot(telecom_hv_cust.max_rech_amt_8[(telecom_hv_cust["churn"] == 0)],

                color="Red", shade = True)

ax = sns.kdeplot(telecom_hv_cust.max_rech_amt_8[(telecom_hv_cust["churn"] == 1)],

                ax =ax, color="Blue", shade= True)

ax.legend(["No-Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('Volume based cost')

ax.set_title('Distribution of Max Recharge Amount by churn')
# we will create a new dataset for model building

df = telecom_hv_cust[:].copy()



# lets drop tenure range because it is highly correlated with AON MONTH column

df.drop('tenure_range', axis=1, inplace=True)

df.drop('mobile_number', axis=1, inplace=True)

df.head()
# lets create X & y dataset for model building, X will obviously not have "churn" and y will only have "churn"

X = df.drop(['churn'], axis=1)

y = df['churn']



df.drop('churn', axis=1, inplace=True)

# apply scaling on the dataset

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)
# split the dateset into train and test datasets

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)

print("Dimension of X_train:", X_train.shape)

print("Dimension of X_test:", X_test.shape)
# As discussed earlier, given dataset is skewed, lets balance the dataset

from imblearn.over_sampling import SMOTE



sm = SMOTE(kind = "regular")

X_tr,y_tr = sm.fit_sample(X_train,y_train)
print("Dimension of X_tr Shape:", X_tr.shape)

print("Dimension of y_tr Shape:", y_tr.shape)



print("Imbalance in Training dataset:",(y_tr != 0).sum()/(y_tr == 0).sum())
# Model Building

# SVM (lets start with linear SVM)

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)



lr = LogisticRegression()



lr.svm = SVC(kernel='linear') 

lr.svm.fit(X_train,y_train)

preds = lr.svm.predict(X_test)

metrics.accuracy_score(y_test, preds)



# linear SVM gave us accuracy of 94% on test data
# we will now using RFE for feature reduction

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

from sklearn.feature_selection import RFE



# lets RFE select 15 most imp features for us

rfe = RFE(lr, 15)   

rfe = rfe.fit(X_tr, y_tr)
rfe_features = list(df.columns[rfe.support_])

print("15 most important features selected by RFE ", rfe_features)
X_rfe = pd.DataFrame(data=X_tr).iloc[:, rfe.support_]

y_rfe = y_tr
# lets create a Logisctic Regression model on the seleted columns by RFE

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=1)

lr.fit(X_rfe, y_rfe)
X_test_rfe = pd.DataFrame(data=X_test).iloc[:, rfe.support_]



y_pred = lr.predict(X_test_rfe)



from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

print('Accuracy on the test dataset:',lr.score(X_test_rfe, y_test))
# lets check classification report on the test dataset

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)



# apply SMOTE to correct class imbalance

from imblearn.over_sampling import SMOTE

sm = SMOTE(kind = "regular")

X_tr,y_tr = sm.fit_sample(X_train,y_train)

print(X_tr.shape)

print(y_tr.shape)

# import PCA

from sklearn.decomposition import PCA

pca = PCA(random_state=100)



# apply PCA on train data

pca.fit(X_tr)
X_tr_pca = pca.fit_transform(X_tr)

print(X_tr_pca.shape)



X_test_pca = pca.transform(X_test)

print(X_test_pca.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

lr_pca = LogisticRegression(C=1e9)

lr_pca.fit(X_tr_pca, y_tr)



# make the predictions

y_pred = lr_pca.predict(X_test_pca)



# convert prediction array into a dataframe

y_pred_df = pd.DataFrame(y_pred)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Printing confusion matrix

print(confusion_matrix(y_test,y_pred))

print("Accuracy of the logistic regression model with PCA: ",accuracy_score(y_test,y_pred))
col = list(df.columns)

df_pca = pd.DataFrame({'PC-1':pca.components_[0],'PC-2':pca.components_[1], 'PC-3':pca.components_[2],'Feature':col})

df_pca.head(10)
# scree plot to check the variance explained by different PCAs

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('no of principal components')

plt.ylabel('explained variance - cumulative')

plt.show()
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# 33 columns explains 90% of the variance, lets apply PCA with 33 components
# PCA with 33 components

pca_33 = PCA(n_components=33)



df_tr_pca_33 = pca_33.fit_transform(X_tr)

print(df_tr_pca_33.shape)



df_test_pca_33 = pca_33.transform(X_test)

print(df_test_pca_33.shape)
# Let's run the model using the selected variables

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

lr_pca1 = LogisticRegression(C=1e9)

lr_pca1.fit(df_tr_pca_33, y_tr)



# Predicted probabilities

y_pred33 = lr_pca1.predict(df_test_pca_33)



# Converting y_pred to a dataframe which is an array

df_y_pred = pd.DataFrame(y_pred33)



print("Accuracy with 33 PCAs: ",accuracy_score(y_test,y_pred33))
print(confusion_matrix(y_test,y_pred33))
# lets create a decision tree now

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)



# apply SMOTE to tackle class imbalance

from imblearn.over_sampling import SMOTE

sm = SMOTE(kind = "regular")

X_tr,y_tr = sm.fit_sample(X_train,y_train)

print(X_tr.shape)

print(y_tr.shape)
# feature selection using lasso

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel

 

svc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(X_tr, y_tr)

svc_model = SelectFromModel(svc, prefit=True)

X_lasso = svc_model.transform(X_tr)

position = svc_model.get_support(indices=True)



print(X_lasso.shape)

print(position)
# feature vector for decision tree

lasso_features = list(df.columns[position])

print("Lasso Features: ", lasso_features)
# import decision tree libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree



# lets create a decision tree with the default hyper parameters except max depth to make the tree readable

dt1 = DecisionTreeClassifier(max_depth=5)

dt1.fit(X_lasso, y_tr)
# lets see the classification reort of the model built

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Model predictions

X_test = pd.DataFrame(data=X_test).iloc[:, position]

y_pred1 = dt1.predict(X_test)



# classification report

print(classification_report(y_test, y_pred1))
# confusion matrix

print(confusion_matrix(y_test,y_pred1))

# accuracy of the decision tree

print('Decision Tree - Accuracy :',accuracy_score(y_test,y_pred1))
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(1, 40)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)

                               

# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                   return_train_score=True)

tree.fit(X_lasso, y_tr)
# grid search results

score = tree.cv_results_

pd.DataFrame(score).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(score["param_max_depth"], 

         score["mean_train_score"], 

         label="training accuracy")

plt.plot(score["param_max_depth"], 

         score["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()





# max_depth =10 seems to be the optimal one
# lets find optimal value of minimum sample leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                   return_train_score=True)

tree.fit(X_lasso, y_tr)
# grid search results

score = tree.cv_results_

pd.DataFrame(score).head()
# plotting accuracies with min_sample_leaf

plt.figure()

plt.plot(score["param_min_samples_leaf"], 

         score["mean_train_score"], 

         label="training accuracy")

plt.plot(score["param_min_samples_leaf"], 

         score["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_sample_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()



# min_sample_leaf =25 seems to be the optimal one
# lets fine tune min sample split now

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(5, 200, 20)}



# instantiate the model

dtree = DecisionTreeClassifier(criterion = "gini", 

                               random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                   return_train_score=True)

tree.fit(X_lasso, y_tr)
# scores of GridSearch CV

scores = tree.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()



# min_samples_leaf=50 seems to be optimal
# Create the parameter grid 

param_grid = {

    'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(25, 175, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]

}



n_folds = 5



# Instantiate the grid search model

dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

                          cv = n_folds, verbose = 1)



# Fit the grid search to the data

grid_search.fit(X_lasso, y_tr)
# cv results

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results

# printing the optimal accuracy score and hyperparameters

print("Best Accuracy", grid_search.best_score_)

print(grid_search.best_estimator_)
# model with optimal hyperparameters

clf_gini = DecisionTreeClassifier(criterion = "gini", 

                                  random_state = 100,

                                  max_depth=5, 

                                  min_samples_leaf=25,

                                  min_samples_split=50)

clf_gini.fit(X_lasso, y_tr)
# accuracy score

print ('Accuracy Score for Decision Tree Final Model :',clf_gini.score(X_test,y_test))
# Conclusion from the above Decision Tree model

# 1. 85% accuracy on the test dataset

# 2. lots of false positives in the confusion matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



model_rf = RandomForestClassifier()

model_rf.fit(X_lasso, y_tr)



# Make predictions

prediction_test = model_rf.predict(X_test)

print ('Randon Forest Accuracy with Default Hyperparameter',metrics.accuracy_score(y_test, prediction_test))
print(classification_report(y_test,prediction_test))
# Printing confusion matrix

print(confusion_matrix(y_test, prediction_test))
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(2, 20, 5)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                  return_train_score=True)

rf.fit(X_lasso, y_tr)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
##Tuning n_estimators

## GridSearchCV to find optimal n_estimators

#from sklearn.model_selection import KFold

## specify number of folds for k-fold CV

n_folds = 5

#

## parameters to build the model on

parameters = {'n_estimators': range(100, 1500, 400)}

#

## instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=4)

#

#

## fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                  return_train_score=True)

rf.fit(X_lasso, y_tr)
## scores of GridSearch CV

scores = rf.cv_results_

#

## plotting accuracies with n_estimators

plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_n_estimators"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(50, 400, 10)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                  return_train_score=True)

rf.fit(X_lasso, y_tr)
# scores of GridSearch CV

scores = rf.cv_results_



# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(100, 500, 25)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy",

                   return_train_score=True)

rf.fit(X_lasso, y_tr)
# scores of GridSearch CV

scores = rf.cv_results_



# plotting accuracies with min_samples_split

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 300, 100),

    'min_samples_split': range(200, 500, 100),

    'n_estimators': [500,700], 

    'max_features': [10,20,25]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
# fit the grid search with the data

grid_search.fit(X_lasso, y_tr)

# optimal accuracy score and hyperparameters

print('Accuracy is',grid_search.best_score_,'using',grid_search.best_params_)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



model_rf = RandomForestClassifier(bootstrap=True,

                                  max_depth=10,

                                  min_samples_leaf=100, 

                                  min_samples_split=200,

                                  n_estimators=1000 ,

                                  oob_score = True, n_jobs = -1,

                                  random_state =50,

                                  max_features = 15,

                                  max_leaf_nodes = 30)

model_rf.fit(X_train, y_train)



# Make predictions

prediction_test = model_rf.predict(X_test)
# evaluation metrics

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,prediction_test))

print(confusion_matrix(y_test,prediction_test))
# accuracy score

print ('Accuracy Score for Random Forest Final Model :',metrics.accuracy_score(y_test, prediction_test))
# list of important features

X = df

features = X.columns.values

X = pd.DataFrame(scaler.transform(X))

X.columns = features



importances = model_rf.feature_importances_

weights = pd.Series(importances,

                 index=X.columns.values)

weights.sort_values()[-10:].plot(kind = 'barh')