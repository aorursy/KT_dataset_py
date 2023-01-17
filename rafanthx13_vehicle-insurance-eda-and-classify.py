import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import eli5 # lib to debug ML Models

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.4f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)
file_path = '/kaggle/input/health-insurance-cross-sell-prediction/train.csv'
df = pd.read_csv(file_path)
print("DataSet = {:,d} rows and {} columns".format(df.shape[0], df.shape[1]))

print("\nAll Columns:\n=>", df.columns.tolist())

quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

print("\nStrings Columns:\n=>", qualitative,
      "\n\nNumerics Columns:\n=>", quantitative)

df.head()
file_path = '/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv'
df_sample_submission = pd.read_csv(file_path)
print("DataSet = {:,d} rows and {} columns".format(df_sample_submission.shape[0], df_sample_submission.shape[1]))
df_sample_submission.head(2)
df.describe().T
# statistics
from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax #for some statistics
from scipy.special import boxcox1p

def test_normal_distribution(serie, series_name='series', thershold=0.4):
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
    f.suptitle('{} is a Normal Distribution?'.format(series_name), fontsize=18)
    ax1.set_title("Histogram to " + series_name)
    ax2.set_title("Q-Q-Plot to "+ series_name)
    # calculate normal distrib. to series
    mu, sigma = norm.fit(serie)
    print('Normal dist. (mu= {:,.2f} and sigma= {:,.2f} )'.format(mu, sigma))
    # skewness and kurtoise
    skewness = serie.skew()
    kurtoise = serie.kurt()
    print("Skewness: {:,.2f} | Kurtosis: {:,.2f}".format(skewness, kurtoise))
    # evaluate skeness
    pre_text = '\t=> '
    if(skewness < 0):
        text = pre_text + 'negatively skewed or left-skewed'
    else:
        text =  pre_text + 'positively skewed or right-skewed\n'
        text += pre_text + 'in case of positive skewness, log transformations usually works well.\n'
        text += pre_text + 'np.log(), np.log1(), boxcox1p()'
    if(skewness < -1 or skewness > 1):
        print("Evaluate skewness: highly skewed")
        print(text)
    if( (skewness <= -0.5 and skewness > -1) or (skewness >= 0.5 and skewness < 1)):
        print("Evaluate skewness: moderately skewed")
        print(text)
    if(skewness >= -0.5 and skewness <= 0.5):
        print('Evaluate skewness: approximately symmetric')
    print('evaluate kurtoise')
    if(kurtoise > 3 + thershold):
        print(pre_text + 'Leptokurtic: anormal: Peak is higher')
    elif(kurtoise < 3 - thershold):
        print(pre_text + 'Platykurtic: anormal: The peak is lower')
    else:
        print(pre_text + 'Mesokurtic: normal: the peack is normal')
    sns.distplot(serie , fit=norm, ax=ax1)
    ax1.legend(['Normal dist. ($\mu=$ {:,.2f} and $\sigma=$ {:,.2f} )'.format(mu, sigma)],
            loc='best')
    ax1.set_ylabel('Frequency')
    stats.probplot(serie, plot=ax2)
    plt.show()
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score

this_labels = ['Response 0','Response 1']

def class_report(y_real, y_my_preds, name="", labels=this_labels):
    if(name != ''):
        print(name,"\n")
    print(confusion_matrix(y_real, y_my_preds), '\n')
    print(classification_report(y_real, y_my_preds, target_names=labels))
def check_balanced_train_test_binary(x_train, y_train, x_test, y_test, original_size, labels):
    train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
    test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)

    prop_train = train_counts_label/ len(y_train)
    prop_test = test_counts_label/ len(y_test)

    print("Original Size:", '{:,d}'.format(original_size))
    print("\nTrain: must be 80% of dataset:\n", 
          "the train dataset has {:,d} rows".format(len(x_train)),
          'this is ({:.2%}) of original dataset'.format(len(x_train)/original_size),
                "\n => Classe 0 ({}):".format(labels[0]), train_counts_label[0], '({:.2%})'.format(prop_train[0]), 
                "\n => Classe 1 ({}):".format(labels[1]), train_counts_label[1], '({:.2%})'.format(prop_train[1]),
          "\n\nTest: must be 20% of dataset:\n",
          "the test dataset has {:,d} rows".format(len(x_test)),
          'this is ({:.2%}) of original dataset'.format(len(x_test)/original_size),
                  "\n => Classe 0 ({}):".format(labels[0]), test_counts_label[0], '({:.2%})'.format(prop_test[0]),
                  "\n => Classe 1 ({}):".format(labels[1]),test_counts_label[1], '({:.2%})'.format(prop_test[1])
         )
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    if(fix_labels):
        val_concat = val_concat.sort_values(series_name).reset_index()
    
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], '{:,d}'.format(int(row['quantity'])), color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
def plot_top_bottom_rank_correlation(my_df, column_target, top_rank=5, title=''):
    corr_matrix = my_df.corr()
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
    if(title):
        f.suptitle(title)

    ax1.set_title('Top {} Positive Corr to {}'.format(top_rank, column_target))
    ax2.set_title('Top {} Negative Corr to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(top_rank+1, column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 8}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    cols_bot = corr_matrix.nsmallest(top_rank, column_target)[column_target].index
    cols_bot  = cols_bot.insert(0, column_target)
    print(cols_bot)
    cm = np.corrcoef(my_df[cols_bot].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10}, yticklabels=cols_bot.values,
                     xticklabels=cols_bot.values, mask=mask, ax=ax2)
    
    plt.show()
def plot_top_rank_correlation(my_df, column_target):
    corr_matrix = my_df.corr()
    top_rank = len(corr_matrix)
    f, ax1 = plt.subplots(ncols=1, figsize=(18, 6), sharex=False)

    ax1.set_title('Top Correlations to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(len(corr_matrix), column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    plt.show()
import time

def time_spent(time0):
    t = time.time() - time0
    t_int = int(t) // 60
    t_min = t % 60
    if(t_int != 0):
        return '{} min {:.3f} s'.format(t_int, t_min)
    else:
        return '{:.3f} s'.format(t_min)
def eda_numerical_feat(series, title="", with_label=True, number_format="", show_describe=False, size_labels=10):
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)
    if(show_describe):
        print(series.describe())
    if(title != ""):
        f.suptitle(title, fontsize=18)
    sns.distplot(series, ax=ax1)
    sns.boxplot(series, ax=ax2)
    if(with_label):
        describe = series.describe()
        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 
              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],
              'Q3': describe.loc['75%']}
        if(number_format != ""):
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',
                         size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
        else:
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
                     size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
    plt.show()
def series_remove_outiliers(series):
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    print('Cut Off: below than', lower, 'and above than', upper)
    outliers = series[ (series > upper) | (series < lower)]
    print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
          '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
    # remove outliers
    outliers_removed = [x for x in series if x >= lower and x <= upper]
    print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
    series_no_outiliers = series[ (series <= upper) & (series >= lower) ]
    return series_no_outiliers
def plot_top_bottom_rank_correlation(my_df, column_target, top_rank=5, title=''):
    corr_matrix = my_df.corr()
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
    if(title):
        f.suptitle(title)

    ax1.set_title('Top {} Positive Corr to {}'.format(top_rank, column_target))
    ax2.set_title('Top {} Negative Corr to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(top_rank+1, column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 11}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    cols_bot = corr_matrix.nsmallest(top_rank, column_target)[column_target].index
    cols_bot  = cols_bot.insert(0, column_target)
    cm = np.corrcoef(my_df[cols_bot].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10}, yticklabels=cols_bot.values,
                     xticklabels=cols_bot.values, mask=mask, ax=ax2)
    
    plt.show()
df.duplicated().sum()
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
eda_categ_feat_desc_plot(df['Response'], ' "Response" Distribution')
eda_categ_feat_desc_plot(df['Gender'], title='gender distribution')
eda_categ_feat_desc_plot(df['Vehicle_Damage'], title='Vehicle_Damage distribution')
eda_categ_feat_desc_plot(df['Vehicle_Age'], title='Vehicle_Age distribution')
eda_categ_feat_desc_plot(df['Driving_License'], title='Driving_License distribution', fix_labels=True)
eda_numerical_feat(df['Age'], title='age distribution', number_format='{:.0f}')
eda_numerical_feat(df['Annual_Premium'], title='Annual_Premium distribution WITH OUTILIERS', number_format='{:,.2f}')
eda_numerical_feat(series_remove_outiliers(df['Annual_Premium']), title='Annual_Premium distribution NO OUTILIERS', number_format='{:,.2f}')
afilter = df["Policy_Sales_Channel"].value_counts().nlargest(30).index.tolist()

fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x=df[ df['Policy_Sales_Channel'].isin(afilter)]["Policy_Sales_Channel"] )
plt.title("Top 30 Policy_Sales_Channel count", fontsize=24)
plt.xlabel('Region_Code', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x="Region_Code", data=df)
plt.title("Number of records per year", fontsize=24)
plt.xlabel('Region_Code', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
eda_categ_feat_desc_plot(df['Previously_Insured'], title='Previously_Insured distribution')
eda_numerical_feat(df['Vintage'], title='Vintage distribution', number_format='{:.0f}')
df1 = df.groupby(['Gender', 'Response']).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)
df1 = df1.drop( df1.columns.tolist()[3::], axis=1)
df1 = df1.rename({df1.columns[2]: 'Quantity'},axis=1)

f, (ax3, ax1, ax2) = plt.subplots(ncols=3, figsize=(20, 5), sharex=False)
f.suptitle('Distribution of Response by Gender', fontsize=18)

sns.countplot(x="Gender", data=df, hue='Response', ax=ax3)
ax3.set_title('CountPlot')

alist = df1['Quantity'].tolist()

df1.query('Gender == "Male"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[2]),
                                                 'Response 1 = {:,d}'.format(alist[3])],
                                       title="Male Responses (Total = {:,d})".format(alist[2] + alist[3]),
                                       ax=ax1, labeldistance=None)

df1.query('Gender == "Female"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[0]),
                                                 'Response 1 = {:,d}'.format(alist[1])],
                                       title="Female Responses (Total = {:,d})".format(alist[0] + alist[1]),
                                       ax=ax2, labeldistance=None)

plt.show()
df1 = df.groupby(['Vehicle_Damage', 'Response']).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)
df1 = df1.drop( df1.columns.tolist()[3::], axis=1)
df1 = df1.rename({df1.columns[2]: 'Quantity'},axis=1)

f, (ax3, ax1, ax2) = plt.subplots(ncols=3, figsize=(20, 5), sharex=False)
f.suptitle('Distribution of Response by Vehicle_Damage', fontsize=18)

sns.countplot(x="Vehicle_Damage", data=df, hue='Response', ax=ax3)
ax3.set_title('CountPlot')

alist = df1['Quantity'].tolist()

df1.query('Vehicle_Damage == "Yes"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[2]),
                                                 'Response 1 = {:,d}'.format(alist[3])],
                                       title="Vehicle_Damage Yes (Total = {:,d})".format(alist[2] + alist[3]),
                                       ax=ax1, labeldistance=None)

df1.query('Vehicle_Damage == "No"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[0]),
                                                 'Response 1 = {:,d}'.format(alist[1])],
                                       title="Vehicle_Damage No (Total = {:,d})".format(alist[0] + alist[1]),
                                       ax=ax2, labeldistance=None)

plt.show()
df1 = df.groupby(['Vehicle_Age', 'Response']).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)
df1 = df1.drop( df1.columns.tolist()[3::], axis=1)
df1 = df1.rename({df1.columns[2]: 'Quantity'},axis=1)

f, (ax3, ax1, ax2, ax4) = plt.subplots(ncols=4, figsize=(20, 5), sharex=False)
f.suptitle('Distribution of Response by Vehicle_Age', fontsize=18)

sns.countplot(x="Vehicle_Age", data=df, hue='Response', ax=ax3)
ax3.set_title('CountPlot')

alist = df1['Quantity'].tolist()

df1.query('Vehicle_Age == "1-2 Year"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[2]),
                                                 'Response 1 = {:,d}'.format(alist[3])],
                                       title="Vehicle_Age 1-2 Year (Total = {:,d})".format(alist[2] + alist[3]),
                                       ax=ax1, labeldistance=None)

df1.query('Vehicle_Age == "< 1 Year"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[0]),
                                                 'Response 1 = {:,d}'.format(alist[1])],
                                       title="Vehicle_Age < 1 Year (Total = {:,d})".format(alist[0] + alist[1]),
                                       ax=ax2, labeldistance=None)

df1.query('Vehicle_Age == "> 2 Years"').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[4]),
                                                 'Response 1 = {:,d}'.format(alist[5])],
                                       title="Vehicle_Age > 2 Years (Total = {:,d})".format(alist[4] + alist[5]),
                                       ax=ax4, labeldistance=None)

plt.show()
df1 = df.groupby(['Driving_License', 'Response']).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)
df1 = df1.drop( df1.columns.tolist()[3::], axis=1)
df1 = df1.rename({df1.columns[2]: 'Quantity'},axis=1)

f, (ax3, ax1, ax2) = plt.subplots(ncols=3, figsize=(20, 5), sharex=False)
f.suptitle('Distribution of Response by Driving_License', fontsize=18)

sns.countplot(x="Driving_License", data=df, hue='Response', ax=ax3)
ax3.set_title('CountPlot')

alist = df1['Quantity'].tolist()

df1.query('Driving_License == 0').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[2]),
                                                 'Response 1 = {:,d}'.format(alist[3])],
                                       title="Driving_License 0 (Total = {:,d})".format(alist[2] + alist[3]),
                                       ax=ax1, labeldistance=None)

df1.query('Driving_License == 1').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[0]),
                                                 'Response 1 = {:,d}'.format(alist[1])],
                                       title="Driving_License 1 (Total = {:,d})".format(alist[0] + alist[1]),
                                       ax=ax2, labeldistance=None)

plt.show()
df1 = df.groupby(['Previously_Insured', 'Response']).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)
df1 = df1.drop( df1.columns.tolist()[3::], axis=1)
df1 = df1.rename({df1.columns[2]: 'Quantity'},axis=1)

f, (ax3, ax1, ax2) = plt.subplots(ncols=3, figsize=(20, 5), sharex=False)
f.suptitle('Distribution of Response by Previously_Insured', fontsize=18)

sns.countplot(x="Previously_Insured", data=df, hue='Response', ax=ax3)
ax3.set_title('CountPlot')

alist = df1['Quantity'].tolist()

df1.query('Previously_Insured == 0').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[2]),
                                                 'Response 1 = {:,d}'.format(alist[3])],
                                       title="Previously_Insured 0 (Total = {:,d})".format(alist[2] + alist[3]),
                                       ax=ax1, labeldistance=None)

df1.query('Previously_Insured == 1').plot.pie(y='Quantity', figsize=(15, 5), autopct='%1.2f%%', 
                                       labels = ['Response 0 = {:,d}'.format(alist[0]),
                                                 'Response 1 = {:,d}'.format(alist[1])],
                                       title="Previously_Insured 1 (Total = {:,d})".format(alist[0] + alist[1]),
                                       ax=ax2, labeldistance=None)

plt.show()
fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x="Region_Code", hue='Response', data=df)
plt.title("Count Responses by Region_code", fontsize=24)
plt.xlabel('Region_Code', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
# Percentage of categorical feauture Region_Code by Response

regions_code = df['Region_Code'].unique()
regions_code

gb = df.groupby(['Region_Code','Response']).count().reset_index()

sum_by_region_code = df['Region_Code'].value_counts().reset_index().rename({'index': 'Region_Codes', 'Region_Code': 'Count'}, axis=1)
sum_by_region_code

adict = {}
for index, row in sum_by_region_code.iterrows():
    index = row['Region_Codes']
    count_0 = int(gb.query('Response == 0 & Region_Code == ' + str(index))['Gender'])
    count_1 = int(gb.query('Response == 1 & Region_Code == ' + str(index))['Gender'])
    adict[index] = [count_0, count_0/row['Count'], count_1, count_1/row['Count'], ]

adict
apandas = pd.DataFrame(adict.values(), columns=['count_0', '%0', 'count_1', '%1'], index=adict.keys()).reset_index()

final_df = sum_by_region_code.merge(apandas,left_on='Region_Codes', right_on='index').drop(['index'], axis=1).sort_values(by='%1', ascending=False)
pd.concat([final_df.head(), final_df.tail()]) # Top 5 fist and bottom sorted by '%1'
afilter = df["Policy_Sales_Channel"].value_counts().nlargest(5).index.tolist()
df_filted = df[ df['Policy_Sales_Channel'].isin(afilter) ] 

fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x='Policy_Sales_Channel', hue='Response', data=df_filted)
plt.title("Top 5 Policy_Sales_Channel count", fontsize=24)
plt.xlabel('Region_Code', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
removes_fists = df["Policy_Sales_Channel"].value_counts().nlargest(5).index.tolist()
afilter = df["Policy_Sales_Channel"].value_counts().nlargest(10).drop(removes_fists).index.tolist()

df_filted = df[ df['Policy_Sales_Channel'].isin(afilter) ] 

fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x='Policy_Sales_Channel', hue='Response', data=df_filted)
plt.title("Top 6-10 Policy_Sales_Channel count", fontsize=24)
plt.xlabel('Region_Code', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
removes_fists = df["Policy_Sales_Channel"].value_counts().nlargest(10).index.tolist()
afilter = df["Policy_Sales_Channel"].value_counts().nlargest(30).drop(removes_fists).index.tolist()

df_filted = df[ df['Policy_Sales_Channel'].isin(afilter) ] 

fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x='Policy_Sales_Channel', hue='Response', data=df_filted)
plt.title("Top 10-30 Policy_Sales_Channel count", fontsize=24)
plt.xlabel('Region_Code', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
gb = df.groupby(['Policy_Sales_Channel','Response']).count().reset_index()

sum_by_region_code = df['Policy_Sales_Channel'].value_counts().reset_index().rename(
    {'index': 'Policy_Sales_Channels', 'Policy_Sales_Channel': 'Count'}, axis=1)

adict = {}
for _, row in sum_by_region_code.iterrows():
    index = row['Policy_Sales_Channels']
    
    query0 = gb.query('Response == 0 & Policy_Sales_Channel == ' + str(index))['Gender']
    count_0 = 0 if len(query0) == 0 else int(query0)
    query1 = gb.query('Response == 1 & Policy_Sales_Channel == ' + str(index))['Gender']
    count_1 = 0 if len(query1) == 0 else int(query1)

    adict[index] = [count_0, count_0/row['Count'], count_1, count_1/row['Count'], ]

apandas = pd.DataFrame(data=adict.values(), index=adict.keys(),
                      columns=['count_0', '%0', 'count_1', '%1'] ).reset_index()

final_df = sum_by_region_code.merge(apandas, left_on='Policy_Sales_Channels', right_on='index').drop(
    ['index'], axis=1).sort_values(by='%1', ascending=False)
pd.concat([final_df.head(), final_df.tail()]) # Top 5 fist and bottom sorted by '%1'
fig, (ax1, ax2) = plt.subplots(figsize = (16,5), ncols=2, sharex=False, sharey=False)

font_size = 14
fig.suptitle('age by reponse', fontsize=18)

sns.boxplot(x="Response", y="Age", data=df, ax=ax1)
sns.distplot(df[(df.Response == 0)]["Age"],color='c',ax=ax2, label='Response 0')
sns.distplot(df[(df.Response == 1)]['Age'],color='b',ax=ax2, label='Response 1')

ax1.set_title('charges by smoke or not', fontsize=font_size)
ax2.set_title('Distribution of charges for smokers or  not', fontsize=font_size)
plt.show()
fig, (ax1, ax2) = plt.subplots(figsize = (16,5), ncols=2, sharex=False, sharey=False)

font_size = 14
fig.suptitle('vintage by reponse', fontsize=18)

sns.boxplot(x="Response", y="Vintage", data=df, ax=ax1)
sns.distplot(df[(df.Response == 0)]["Vintage"],color='c',ax=ax2, label='Response 0')
sns.distplot(df[(df.Response == 1)]['Vintage'],color='b',ax=ax2, label='Response 1')

ax1.set_title('charges by smoke or not', fontsize=font_size)
ax2.set_title('Distribution of charges for smokers or  not', fontsize=font_size)
plt.show()
df2 = df.copy()
df2['Annual_Premium'] = series_remove_outiliers(df['Annual_Premium'])

fig, (ax1, ax2) = plt.subplots(figsize = (16,5), ncols=2, sharex=False, sharey=False)

font_size = 14
fig.suptitle('Annual_Premium by reponse', fontsize=18)

sns.boxplot(x="Response", y="Annual_Premium", data=df2, ax=ax1)
sns.distplot(df2[(df2.Response == 0)]["Annual_Premium"],color='c',ax=ax2, label='Response 0')
sns.distplot(df2[(df2.Response == 1)]['Annual_Premium'],color='b',ax=ax2, label='Response 1')

ax1.set_title('charges by smoke or not', fontsize=font_size)
ax2.set_title('Distribution of charges for smokers or  not', fontsize=font_size)
plt.show()
df1 = df.copy()

gender = {'Male': 0, 'Female': 1}
driving_license = {0: 0, 1: 1}
previously_insured = {0: 1, 1: 0}
vehicle_age = {'> 2 Years': 3, '1-2 Year': 2, '< 1 Year': 1}
vehicle_damage = {'Yes': 1, 'No': 0}

df1['Gender'] = df1['Gender'].replace(gender)
df1['Driving_License'] = df1['Driving_License'].replace(driving_license)
df1['Previously_Insured'] = df1['Previously_Insured'].replace(previously_insured)
df1['Vehicle_Age'] = df1['Vehicle_Age'].replace(vehicle_age)
df1['Vehicle_Damage'] = df1['Vehicle_Damage'].replace(vehicle_damage)
df1 = df1.drop(['id'],axis=1)

df2, df3 = df1.copy(), df1.copy()
df1['Policy_Sales_Channel'] = df1['Policy_Sales_Channel'].astype('int32').astype('object')
df1 = pd.concat([df1, pd.get_dummies(df['Policy_Sales_Channel'], prefix='PSC')], axis=1)
df1.iloc[:, 10:25].head(3) # 15 columns of Policy_Sales_Channel as OneHotEncoding
plot_top_bottom_rank_correlation(df1, 'Response', top_rank=12, title='Corr to PolicySalesChanell values')
df2['Region_Code'] = df2['Region_Code'].astype('int32').astype('object')
df2 = pd.concat([df2, pd.get_dummies(df['Region_Code'], prefix='RC')], axis=1)
df2.iloc[:, 10:25].head(3) # 15 columns of RegionCode as OneHotEncoding
plot_top_bottom_rank_correlation(df2, 'Response', top_rank=10, title='Corr to Region_Code values')
plot_top_rank_correlation(df3, 'Response')
test_normal_distribution(df['Annual_Premium'], 'Annual_Premium')
# df['Annual_Premium'] = boxcox1p(df['Annual_Premium'], boxcox_normmax(df['Annual_Premium'] + 1))
df['Annual_Premium'] = df['Annual_Premium'].apply(np.log)
test_normal_distribution(df['Annual_Premium'], 'Annual_Premium')
gender = {'Male': 0, 'Female': 1}
driving_license = {0: 0, 1: 1}
previously_insured = {0: 1, 1: 0}
vehicle_age = {'> 2 Years': 3, '1-2 Year': 2, '< 1 Year': 1}
vehicle_damage = {'Yes': 1, 'No': 0}

df['Gender'] = df['Gender'].replace(gender)
df['Driving_License'] = df['Driving_License'].replace(driving_license)
df['Previously_Insured'] = df['Previously_Insured'].replace(previously_insured)
df['Vehicle_Age'] = df['Vehicle_Age'].replace(vehicle_age)
df['Vehicle_Damage'] = df['Vehicle_Damage'].replace(vehicle_damage)

df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].apply(lambda x: np.int(x))
df['Region_Code'] = df['Region_Code'].apply(lambda x: np.int(x))

df = df.drop(['id'],axis=1)
df.head()
# OnetHotEndonding to Bigs Coor of 'Policy_Sales_Channel' and 'Region_Code'
# OneHotEnconding to 'Policy_Sales_Channel' and 'Region_Code'

# df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int32').astype('object')
# df['Region_Code'] = df['Region_Code'].astype('int32').astype('object')
# df = pd.concat([df, 
#                 pd.get_dummies(df['Region_Code'], prefix='RC'),
#                 pd.get_dummies(df['Policy_Sales_Channel'], prefix='PSC')], axis=1)
# df # df with PSC and RC dummies
from sklearn.model_selection import train_test_split

col_1 = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
         'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

cat_col = ['Gender','Driving_License', 'Region_Code', 'Previously_Insured',
           'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']

X = df[col_1]
# X = df.drop(['Response'], axis=1)

y = df['Response']

x_train, x_test, y_train, y_test = train_test_split(X, y.values, test_size=0.20, random_state=42)

check_balanced_train_test_binary(x_train, y_train, x_test, y_test, len(df), ['Response 0', 'Response 1'])
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek # over and under sampling
from imblearn.metrics import classification_report_imbalanced

imb_models = {
    'ADASYN': ADASYN(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'SMOTEENN': SMOTEENN("minority", random_state=42),
    'SMOTETomek': SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42)
}

imb_strategy = "None"

if(imb_strategy != "None"):
    before = x_train.shape[0]
    imb_tranformer = imb_models[imb_strategy]
    x_train, y_train = imb_tranformer.fit_sample(x_train, y_train)
    print("train dataset before: {:,d}\nimbalanced_strategy: {}".format(before, imb_strategy),
          "\ntrain dataset after: {:,d}\ngenerate: {:,d}".format(x_train.shape[0], x_train.shape[0] - before))
else:
    print("Dont correct unbalanced dataset")
# Classifier Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensemble Classifiers
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Others Linear Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier

# xboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# scores
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# neural net of sklearn
from sklearn.neural_network import MLPClassifier

# others
import time
import operator
all_classifiers = {
    "NaiveBayes": GaussianNB(),
#     "Ridge": RidgeClassifier(),
#     "Perceptron": Perceptron(),
#     "PassiveAggr": PassiveAggressiveClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGB": LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1),
    # "SVM": SVC(),
    "LogisiticR": LogisticRegression(),
#     "KNearest": KNeighborsClassifier(),
#     "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(), # All 100 features: 48min
    # "SGDC": SGDClassifier(),
    "GBoost": GradientBoostingClassifier(),
#     "Bagging": BaggingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ExtraTree": ExtraTreesClassifier()
}
metrics = { 'cv_roc': {}, 'acc_test': {}, 'f1_test': {}, 'roc_auc': {} }
m = list(metrics.keys())
time_start = time.time()
print('CrossValidation, Fitting and Testing')

# Cross Validation, Fit and Test
for name, model in all_classifiers.items():
    print('{:15}'.format(name), end='')
    t0 = time.time()
    # Cross Validation
    training_score = cross_val_score(model, x_train, y_train, scoring="roc_auc", cv=4)
    # Fitting
    all_classifiers[name] = model.fit(x_train, y_train) 
    # Testing
    y_pred = all_classifiers[name].predict(x_test)
    t1 = time.time()
    # Save metrics
    metrics[m[0]][name] = training_score.mean()
    metrics[m[1]][name] = accuracy_score(y_test, y_pred)
    metrics[m[2]][name] = f1_score(y_test, y_pred, average="macro") 
    metrics[m[3]][name] = roc_auc_score(y_test, all_classifiers[name].predict_proba(x_test)[:, 1])
    # Show metrics
    print('| {}: {:6,.4f} | {}: {:6,.4f} | {}: {:6.4f} | {}: {:6.4f} | took: {:>15} |'.format(
        m[0], metrics[m[0]][name], m[1], metrics[m[1]][name],
        m[2], metrics[m[2]][name], m[3], metrics[m[3]][name], time_spent(t0) ))
        
print("\nDone in {}".format(time_spent(time_start)))
print("Best cv acc  :", max( metrics[m[0]].items(), key=operator.itemgetter(1) ))
print("Best acc test:", max( metrics[m[1]].items(), key=operator.itemgetter(1) ))
print("Best f1 test :", max( metrics[m[2]].items(), key=operator.itemgetter(1) ))
print("Best roc_auc :", max( metrics[m[3]].items(), key=operator.itemgetter(1) ))

df_metrics = pd.DataFrame(data = [list(metrics[m[0]].values()),
                                  list(metrics[m[1]].values()),
                                  list(metrics[m[2]].values()),
                                  list(metrics[m[3]].values())],
                          index = ['cv_acc', 'acc_test', 'f1_test', 'roc_auc'],
                          columns = metrics[m[0]].keys() ).T.sort_values(by=m[3], ascending=False)
df_metrics
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

name = 'CatBoost'
catb = CatBoostClassifier()
t0 = time.time()
# Fitting
# catb = catb.fit(x_train, y_train, eval_set=(x_test, y_test), plot=False, early_stopping_rounds=30,verbose=0)
catb = catb.fit(x_train, y_train, cat_features=cat_col, eval_set=(x_test, y_test), plot=False, early_stopping_rounds=30,verbose=0) 
# Testing
y_pred = catb.predict(x_test)
t1 = time.time()
# Save metrics
metrics[m[0]][name] = 0.0
metrics[m[1]][name] = accuracy_score(y_test, y_pred)
metrics[m[2]][name] = f1_score(y_test, y_pred, average="macro") 
metrics[m[3]][name] = roc_auc_score(y_test, catb.predict_proba(x_test)[:, 1]) 
# Show metrics
print('{:15} | {}: {:6,.4f} | {}: {:6.4f} | {}: {:6.4f} | took: {:>15} |'.format(
    name, m[1], metrics[m[1]][name],
    m[2], metrics[m[2]][name], m[3], metrics[m[3]][name], time_spent(t0) ))
feat_importances = pd.Series(catb.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
eli5.explain_weights(catb)
# https://eli5.readthedocs.io/en/latest/

from eli5.sklearn import PermutationImportance

# Check for Permutation Importance of Features
perm = PermutationImportance(all_classifiers['LightGB'], random_state=42).fit(x_test, y_test)
eli5.show_weights(perm, feature_names=X.columns.tolist())
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from mlens.ensemble import SuperLearner
 
# create a list of base-models
def get_models():
    models = list()
    models.append(LogisticRegression(solver='liblinear'))
    models.append(DecisionTreeClassifier())
    #     models.append(SVC(gamma='scale', probability=True))
    models.append(GaussianNB())
    #     models.append(KNeighborsClassifier())
    models.append(XGBClassifier())
    models.append(AdaBoostClassifier())
    #     models.append(BaggingClassifier(n_estimators=10))
    models.append(RandomForestClassifier())
    models.append(LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=42))
    return models
 
# create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=accuracy_score, folds=5, shuffle=True, sample_size=len(X))
    # add base models
    models = get_models()
    ensemble.add(models)
    # add the meta model
    ensemble.add_meta(LogisticRegression(solver='lbfgs'))
    return ensemble
t0 = time.time()

# create the super learner
ensemble = get_super_learner(x_train.values)

# fit the super learner
ensemble.fit(x_train.values, y_train)

# summarize base learners
print(ensemble.data)

# make predictions on hold out set
y_pred = ensemble.predict(x_test.values)

print("took ", time_spent(t0))
class_report(y_test, y_pred, name="SuperLeaner")

y_probs = ensemble.predict_proba(x_test.values)

roc_auc_score(y_test, y_probs)
file_path = '/kaggle/input/health-insurance-cross-sell-prediction/test.csv'
df_test = pd.read_csv(file_path)
df_test.head(2)
df_test['Policy_Sales_Channel'] = df_test['Policy_Sales_Channel'].astype('int32').astype('object')
df_test['Region_Code'] = df_test['Region_Code'].astype('int32').astype('object')
# df_test = pd.concat([df_test, 
#                 pd.get_dummies(df_test['Region_Code'], prefix='RC'),
#                 pd.get_dummies(df_test['Policy_Sales_Channel'], prefix='PSC')], axis=1)
df_test.head()
file_path = '/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv'
df_sample_submission = pd.read_csv(file_path)
df_sample_submission.head(2)
## WITH ALL FEATURES: PRE-PROCESSING
# extend_columns = df.columns.tolist()[11:]
# rc = extend_columns[:53]
# psc = extend_columns[53:]

# for c in extend_columns:
#     df_test[c] = 0
    
# for c in rc:
#     num = int(c[3:])
#     df_test[c] = df_test['Region_Code'].apply(lambda x: 1 if x == num else 0)
    
# for c in psc:
#     num = int(c[4:])
#     df_test[c] = df_test['Policy_Sales_Channel'].apply(lambda x: 1 if x == num else 0)
gender = {'Male': 0, 'Female': 1}
driving_license = {0: 0, 1: 1}
previously_insured = {0: 1, 1: 0}
vehicle_age = {'> 2 Years': 3, '1-2 Year': 2, '< 1 Year': 1}
vehicle_damage = {'Yes': 1, 'No': 0}

df_test['Gender'] = df_test['Gender'].replace(gender)
df_test['Driving_License'] = df_test['Driving_License'].replace(driving_license)
df_test['Previously_Insured'] = df_test['Previously_Insured'].replace(previously_insured)
df_test['Vehicle_Age'] = df_test['Vehicle_Age'].replace(vehicle_age)
df_test['Vehicle_Damage'] = df_test['Vehicle_Damage'].replace(vehicle_damage)

df_test['Policy_Sales_Channel'] = df_test['Policy_Sales_Channel'].apply(lambda x: np.int(x))
df_test['Region_Code'] = df_test['Region_Code'].apply(lambda x: np.int(x))

# df_test['Annual_Premium'] = boxcox1p(df_test['Annual_Premium'], boxcox_normmax(df_test['Annual_Premium'] + 1))
df_test['Annual_Premium'] = df_test['Annual_Premium'].apply(np.log)

df_test.head()
X_sub = df_test.drop(['id'],axis=1)
X_sub.head(2)
all_classifiers.keys()
# all_classifiers['LightGB'] | catb
sub_pred = all_classifiers['XGBoost'].predict_proba(X_sub)[:, 1]
df_sample_submission['Response'] = sub_pred
df_sample_submission.to_csv("XGBoost_simples_aproach.csv", index = False) 
