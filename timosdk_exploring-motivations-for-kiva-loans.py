PUBLISH = True

#import graphing and data frame libs
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from wordcloud import WordCloud
from mpl_toolkits.basemap import Basemap

# statistics :)
from scipy.stats import ttest_ind, probplot

# we will do a little machine learning here - it is kaggle after all
from sklearn.ensemble import RandomForestClassifier

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#pd.options.display.max_rows = 999

%matplotlib inline
%config InlineBackend.figure_format='retina'
plt.rcParams['figure.figsize'] = [14.0, 6.0]
plt.rcParams.update({'font.size': 14})

# load the data
kiva_loans = pd.read_csv("../input/kiva_loans.csv")

# dont trust the column metrics that they give you - there are null values hiding in here!
kiva_loans.isnull().sum()
plt.figure(figsize=[14,14])

plt.subplot(211)
plt.hist(x = kiva_loans.loc[kiva_loans['loan_amount'] < 50000, 'loan_amount'], 
         color = ['#3CB371'], bins = 499)
plt.title('Loan amount histogram for loans less than $50,000 (log y scale)')
plt.xlabel('Loan amount ($)')
plt.ylabel('Number of people applying (log scale)')
plt.yscale('log', nonposy='clip')

plt.subplot(212)
plt.hist(x = kiva_loans.loc[kiva_loans['loan_amount'] < 10000, 'loan_amount'], 
         color = ['#3CB371'], bins = 99)
plt.title('Loan amount histogram for loans less than $5,000')
plt.xlabel('Loan amount ($)')
plt.ylabel('Number of people applying')

sns.despine()
print ("Mean loan amount is ${}, median ${}, mode ${}".format(kiva_loans['loan_amount'].mean(), kiva_loans['loan_amount'].median(), kiva_loans['loan_amount'].mode()[0]))
fig = plt.figure()
ax1 = fig.add_subplot(121)
probplot(kiva_loans['loan_amount'], plot=ax1)

ax2 = fig.add_subplot(122)
probplot(kiva_loans['loan_amount'], plot=ax2)
ax2.set_yscale('log', nonposy='clip')
ax2.set_ylim([10, 100000])


kiva_loans['ask_fund_delta'] = kiva_loans['loan_amount'] - kiva_loans['funded_amount']
#kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'ask_fund_delta']
kiva_loans['fully_unfunded'] = (kiva_loans['loan_amount'] == kiva_loans['ask_fund_delta'])
kiva_loans['partially_unfunded'] = (kiva_loans['ask_fund_delta'] > 0)

no_of_partially_funded = len(kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'ask_fund_delta'])
no_of_fully_unfunded   = kiva_loans['fully_unfunded'].sum()

print ("{:,} people have non- or partially funded loans ({:.1f}% of total loans). \
Of these {:,} are fully unfunded ({:.1f}% of partial, {:.1f}% of total)\
".format(no_of_partially_funded, 
         no_of_partially_funded/len(kiva_loans)*100.0,
         no_of_fully_unfunded,
         no_of_fully_unfunded/no_of_partially_funded*100.0,
         no_of_fully_unfunded/len(kiva_loans)*100.0,
        )
      )

plt.hist(x = kiva_loans.loc[kiva_loans['loan_amount'] < 50000, 'loan_amount'], 
         color = ['#FF6347'], bins = 500)

plt.hist(x = kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'ask_fund_delta'], 
         color = ['#3CB371'], bins = 500)

plt.title('Difference in ask and funded amount')
plt.xlabel('Difference ($)')
plt.ylabel('Number of people affected (log scale)')
plt.yscale('log', nonposy='clip')

sns.despine()
kiva_loans.index = pd.to_datetime(kiva_loans['posted_time'])
ax = kiva_loans['ask_fund_delta'].resample('w').sum().plot()
ax = kiva_loans['loan_amount'].resample('w').sum().plot()
ax.set_ylabel('Amount ($)')
ax.set_xlabel('')
ax.set_xlim((pd.to_datetime(kiva_loans['posted_time'].min()), 
             pd.to_datetime(kiva_loans['posted_time'].max())))
ax.legend(["loan unfunded", "loan ask"])
plt.title('Loan ask and loan unfunded over time')

sns.despine()
plt.show()
# sanity check - above shows $3 mio per week is average ask. Is that reasonable? 
# use timeperiod of 3 years and 7 months - could probably figure out exactly how many
# weeks are covered in this dataset - but this is just a back of the envelope check.
print(" *"*30)
print("Sanity check: total loan amount asked for over time period was ${:,.0f}\
 - that is approximately ${:,.0f} per week.\
".format(kiva_loans['loan_amount'].sum(),
         kiva_loans['loan_amount'].sum()/((3+7.0/12.0) * 52)
        )
     )
print(" *"*30)
# Calculate percent female on loan - works because counting word male also counts female :)
kiva_loans['percent_female'] = kiva_loans['borrower_genders'].str.count('female') / \
                               kiva_loans['borrower_genders'].str.count('male')

kiva_loans['team_gender'] = 'mixed'
kiva_loans.loc[kiva_loans['percent_female'] == 1, 'team_gender'] = 'female'
kiva_loans.loc[kiva_loans['percent_female'] == 0, 'team_gender'] = 'male'
#kiva_loans['team_gender'].value_counts()


# now create training sub set

# drop all the nans
kiva_train = kiva_loans[['partially_unfunded', 'loan_amount', 'date', 'percent_female',
                        'sector', 'country', 'ask_fund_delta', 'repayment_interval']].dropna(axis=0, how='any')
print ("After dropna we still have {:,} of the {:,} partially unfunded\
".format(kiva_train['partially_unfunded'].sum(), kiva_loans['partially_unfunded'].sum()))

# limit the loan amount
kiva_train = kiva_train.drop(kiva_train[kiva_train.loan_amount > 10000].index)
print ("After loan limitation we still have {:,} of the {:,} partially unfunded\
".format(kiva_train['partially_unfunded'].sum(), kiva_loans['partially_unfunded'].sum()))

# limit by date to avoid loading up on partially unfunded loans that just 
# did not have enough time to get funded
kiva_train = kiva_train.drop(kiva_train[kiva_train.date >= '2017-05-01'].index)
print ("After date limitation we still have {:,} of the {:,} partially unfunded\
".format(kiva_train['partially_unfunded'].sum(), kiva_loans['partially_unfunded'].sum()))

# first explore the training set to see if we can see obvious differences 
# between funded and non funded
fig, (maxis1, maxis2) = plt.subplots(1, 2)

maxis1.set_title("Loan amount")
maxis2.set_title("Percent female")

sns.boxplot(x="partially_unfunded", y="loan_amount", data=kiva_train, 
            ax = maxis1, showmeans = True, meanline = True)
sns.boxplot(x="partially_unfunded", y="percent_female", data=kiva_train, 
            ax = maxis2, showmeans = True, meanline = True)

sns.despine()
plt.show()
# Definitely looks like there are some significant differences there. 
# Now lets look at countries.
fig, (maxis1, maxis2) = plt.subplots(2, 1, figsize=[14,12])

maxis1.set_title("Funded loans - top countries")
maxis2.set_title("Partially funded loans - top countries")

sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == False].country.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == False].country.value_counts().head(10), ax = maxis1)

sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == True].country.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == True].country.value_counts().head(10), ax = maxis2)

maxis1.set_ylabel('Number of funded loans')
maxis2.set_ylabel('Number of partially funded loans')

sns.despine()
plt.show()
# Same for sector
fig, (maxis1, maxis2) = plt.subplots(2, 1, figsize=[14,14])

maxis1.set_title("Funded loans - top sectors")
maxis2.set_title("Partially funded loans - top sectors")
sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == False].sector.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == False].sector.value_counts().head(10), ax = maxis1)

sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == True].sector.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == True].sector.value_counts().head(10), ax = maxis2)

maxis1.set_ylabel('Number of funded loans')
maxis2.set_ylabel('Number of partially funded loans')

for tick in maxis1.get_xticklabels():
    tick.set_rotation(10)

for tick in maxis2.get_xticklabels():
    tick.set_rotation(10)

sns.despine()
plt.show()
mostfrequentcountries = kiva_train['country'].value_counts().nlargest(20).keys()
kiva_train.loc[(kiva_train['country'].isin(mostfrequentcountries)==False), 'country'] = "non-top20-country"

# here is the value counts for the full dataset
print ("Country frequencies\n", kiva_train['country'].value_counts())
# now create dummies from sector and country
kiva_train_final = pd.concat([pd.get_dummies(kiva_train['country']), 
                              pd.get_dummies(kiva_train['sector']),
                              pd.get_dummies(kiva_train['repayment_interval']),
                             ], axis = 1)

kiva_train_final['loan_amount'] = kiva_train['loan_amount']
kiva_train_final['percent_female'] = kiva_train['percent_female']

kiva_train_final.sample(5)
np_train_features = kiva_train_final.as_matrix()
print ("training features shape", np_train_features.shape)

np_train_labels = kiva_train['partially_unfunded'].astype(int)
print ("training labels shape", np_train_labels.shape)

features = kiva_train_final.columns
# this one is slow - only run it when we are creating the final kernel
if PUBLISH:
    rfc = RandomForestClassifier(n_estimators=50, min_samples_split=4)

    rfc.fit(np_train_features, np_train_labels)
    score = rfc.score(np_train_features, np_train_labels)

    print("Accuracy on full set: {:0.2f}".format(score*100))
    print(" *"*25)
    print("Top 20 feature importances in this model:")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print("%0.2f%% %s" % (importances[indices[f]]*100, features[indices[f]]))
print("Percent loans requested by only female teams {:.2f}%\
".format(kiva_loans['team_gender'].value_counts()['female']/len(kiva_loans)*100  ))

print("Percent loans requested by only male teams {:.2f}%\
".format(kiva_loans['team_gender'].value_counts()['male']/len(kiva_loans)*100  ))
ax=sns.kdeplot(kiva_loans['percent_female'], color='#3CB371',shade=True, label='all borrowers', bw=0.02)

ax=sns.kdeplot(kiva_loans.loc[(kiva_loans['percent_female'] > 0) & 
                              (kiva_loans['percent_female'] < 1), 'percent_female'], 
               color='#FF6347', shade=True, label='mixed gender teams', bw=0.02)

ax.annotate("0.33",
            xy=(0.33, 1), xycoords='data',
            xytext=(0.33, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

ax.annotate("0.67",
            xy=(0.67, 2.5), xycoords='data',
            xytext=(0.67, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

ax.annotate("0.5",
            xy=(0.5, 3), xycoords='data',
            xytext=(0.5, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

sns.despine()
plt.show()
# plot these as a facet grid...
g = sns.FacetGrid(kiva_loans, col = 'sector', col_wrap=4)
g.map(sns.kdeplot, 'percent_female', color='#3CB371', shade=True, label='mixed gender teams', bw=0.02)
plt.show()
kiva_loans[(kiva_loans['sector'] == 'Personal Use') & 
           (kiva_loans['percent_female'] > 0) &
           (kiva_loans['percent_female'] < 1)
          ].sample(20).use
kiva_loans[(kiva_loans['use'] == 'to buy a sound system for her house.')]
labels1 = 'Funded', 'Partially funded'
sizes1  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 0)
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 0)
          ])]

labels2 = 'Funded', 'Partially funded'
sizes2  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 1)
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 1)
          ])]

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=-45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis1.axis('equal')

maxis2.pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=False, startangle=45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis2.axis('equal')

maxis1.set_title("Male only")
maxis2.set_title("Female only")

maxis1.add_patch(
    patches.Arrow(
        0.1, 0.1,
        1.0, 1.0,
        color = '#3CB371'
    )
)


maxis2.add_patch(
    patches.Rectangle(
        (-0.15, -1.5),   # (x,y)
        0.3,          # width
        0.8,          # height
        facecolor='#3CB371'
    )
)
maxis2.add_patch(
    patches.Rectangle(
        (-0.4, -1.2),   # (x,y)
        0.8,          # width
        0.15,          # height
        facecolor='#3CB371'
    )
)

plt.show()
male = kiva_train[kiva_train['percent_female']==0]
female = kiva_train[kiva_train['percent_female']==1]
ttest = ttest_ind(male['loan_amount'], female['loan_amount'])

print("Mean loan amount for males is ${:,.2f} and for females ${:,.2f}. \
A t-test comparison find these are different with a p-value of {:,.4}.\
\n".format(male['loan_amount'].mean(),
         female['loan_amount'].mean(),
         ttest.pvalue
        ))


male = kiva_train[(kiva_train['percent_female']==0) & (kiva_train['sector'] == 'Construction')]
female = kiva_train[(kiva_train['percent_female']==1) & (kiva_train['sector'] == 'Construction')]
ttest = ttest_ind(male['loan_amount'], female['loan_amount'])

print("Mean loan amount for male construction projects is ${:,.2f} and for female construction projects ${:,.2f}. \
A t-test comparison find these are different with a p-value of {:,.4}.\
\n".format(male['loan_amount'].mean(),
         female['loan_amount'].mean(),
         ttest.pvalue
        ))
labels1 = 'Funded', 'Partially funded'
sizes1  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 0) & (kiva_train['sector'] == 'Construction')
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 0) & (kiva_train['sector'] == 'Construction')
          ])]

labels2 = 'Funded', 'Partially funded'
sizes2  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 1) & (kiva_train['sector'] == 'Construction')
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 1) & (kiva_train['sector'] == 'Construction')
          ])]

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=-45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis1.axis('equal')

maxis2.pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=False, startangle=45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis2.axis('equal')

maxis1.set_title("Male Construction Projects")
maxis2.set_title("Female Construction Projects")

maxis1.add_patch(
    patches.Arrow(
        0.1, 0.1,
        1.0, 1.0,
        color = '#3CB371'
    )
)

maxis2.add_patch(
    patches.Rectangle(
        (-0.15, -1.5),   # (x,y)
        0.3,          # width
        0.8,          # height
        facecolor='#3CB371'
    )
)
maxis2.add_patch(
    patches.Rectangle(
        (-0.4, -1.2),   # (x,y)
        0.8,          # width
        0.15,          # height
        facecolor='#3CB371'
    )
)

plt.show()

# convert the dates to datetime so we can easily manipulate them
kiva_loans['posted_time'] = pd.to_datetime(kiva_loans['posted_time'])
kiva_loans['funded_time'] = pd.to_datetime(kiva_loans['funded_time'])
kiva_loans['disbursed_time'] = pd.to_datetime(kiva_loans['disbursed_time'])

posttofund = kiva_loans['funded_time'] - kiva_loans['posted_time']
posttodisburse = kiva_loans['disbursed_time'] - kiva_loans['posted_time']
fundtodisburse = kiva_loans['disbursed_time'] - kiva_loans['funded_time']

kiva_loans['posted_to_funded_time_in_hours'] = posttofund.dt.components.hours + (posttofund.dt.days*24)
if PUBLISH:
    kiva_loans['posted_to_disbursed_time_in_hours'] = posttodisburse.dt.components.hours + (posttodisburse.dt.days*24)
    kiva_loans['funded_to_disbursed_time_in_hours'] = fundtodisburse.dt.components.hours + (fundtodisburse.dt.days*24)
plt.figure(figsize=[14,13])
maxis1 = plt.subplot(211)
maxis2 = plt.subplot(223)
maxis3 = plt.subplot(224)

# Full time
sns.kdeplot(kiva_loans['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='', bw=12, ax = maxis1)

# First two months
sns.kdeplot(kiva_loans[kiva_loans['posted_to_funded_time_in_hours'] < 1488]['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='', bw=12, ax = maxis2)
sns.kdeplot(kiva_loans[(kiva_loans['posted_to_funded_time_in_hours'] < 1488)& 
                                  (kiva_loans['percent_female'] == 0)]['posted_to_funded_time_in_hours'], 
               color='#5DADE2',shade=False, label='male', bw=12, ax = maxis2)
sns.kdeplot(kiva_loans[(kiva_loans['posted_to_funded_time_in_hours'] < 1488)& 
                                  (kiva_loans['percent_female'] == 1)]['posted_to_funded_time_in_hours'], 
               color='#FF6347',shade=False, label='female', bw=12, ax = maxis2)
maxis2.set_xlim([0,1500])

# First week
sns.kdeplot(kiva_loans[(kiva_loans['posted_to_funded_time_in_hours'] < 168)]['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='', bw=12, ax = maxis3)
maxis3.set_xlim([0,168])

maxis1.set_title("All time")
maxis2.set_title("First two months")
maxis3.set_title("First week")

maxis1.set_xlabel('Time from posting to funding (hours)')
maxis2.set_xlabel('Time from posting to funding (hours)')
maxis3.set_xlabel('Time from posting to funding (hours)')

maxis1.set_yticks([])
maxis1.set_yticklabels([])
maxis2.set_yticks([])
maxis2.set_yticklabels([])
maxis3.set_yticks([])
maxis3.set_yticklabels([])

maxis2.axvline(x=168, color = 'black', lw = 1) 
maxis2.annotate("One week",
            xy=(168, 0), xycoords='data',
            xytext=(174, 0.00025), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

maxis2.axvline(x=774, color = 'black', lw = 1) 
maxis2.annotate("One month",
            xy=(774, 0), xycoords='data',
            xytext=(780, 0.00025), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

sns.despine()
plt.show()
kiva_loans['funded_in_first_two_weeks'] = 0
kiva_loans.loc[kiva_loans['posted_to_funded_time_in_hours'] < 24*14, 'funded_in_first_two_weeks'] = 1
labels = 'Funded in two weeks', 'Not funded in two weeks'
sizes1  = [len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == True) &
           (kiva_loans['percent_female'] == 0)
          ]), len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == False) &
           (kiva_loans['percent_female'] == 0)
          ])]

sizes2  = [len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == True) &
           (kiva_loans['percent_female'] == 1)
          ]), len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == False) &
           (kiva_loans['percent_female'] == 1)
          ])]

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.pie(sizes1, labels=labels, autopct='%1.1f%%', explode = (0, 0.02), 
        shadow=False, startangle=-30, colors = ['#3CB371', '#FF6347'])
maxis1.axis('equal')

maxis2.pie(sizes2, labels=labels, autopct='%1.1f%%', explode = (0, 0.02), 
        shadow=False, startangle=155, colors = ['#3CB371', '#FF6347'])
maxis2.axis('equal')

maxis1.set_title("Male only")
maxis2.set_title("Female only")

maxis1.add_patch(
    patches.Arrow(
        0.1, 0.1,
        1.0, 1.0,
        color = '#3CB371'
    )
)


maxis2.add_patch(
    patches.Rectangle(
        (-0.15, -1.5),   # (x,y)
        0.3,          # width
        0.8,          # height
        facecolor='#3CB371'
    )
)
maxis2.add_patch(
    patches.Rectangle(
        (-0.4, -1.2),   # (x,y)
        0.8,          # width
        0.15,          # height
        facecolor='#3CB371'
    )
)

plt.show()
if PUBLISH:
    sns.kdeplot(kiva_loans['funded_to_disbursed_time_in_hours'], 
                color='#5DADE2',shade=True, label='funded to disbursed', bw=12)

    sns.kdeplot(kiva_loans['posted_to_disbursed_time_in_hours'], 
               color='#FF6347',shade=True, label='posted to disbursed', bw=12)

sns.kdeplot(kiva_loans['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='posted to funded', bw=12)

plt.xlabel('Time (hours)')

# Sanity check
if PUBLISH:
    avr_post = (kiva_loans.posted_time - kiva_loans.posted_time.min()).mean() + kiva_loans.posted_time.min()
    avr_fund = (kiva_loans.funded_time - kiva_loans.funded_time.min()).mean() + kiva_loans.funded_time.min()
    avr_disb = (kiva_loans.disbursed_time - kiva_loans.disbursed_time.min()).mean() + kiva_loans.disbursed_time.min()
    print ("average posting time {}, funding time {}, and disbural time {}.".format(avr_post, avr_fund, avr_disb))

sns.despine()
fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("All lender counts")
maxis2.set_title("Focus on lender counts < 100")

sns.kdeplot(kiva_loans[(kiva_loans['lender_count'] > 0)].lender_count, 
               color='#3CB371',shade=True, label='number of lenders', bw=4, ax=maxis1)

sns.kdeplot(kiva_loans[(kiva_loans['lender_count'] <100) & (kiva_loans['lender_count'] > 0)].lender_count, 
               color='#3CB371',shade=True, label='number of lenders', bw=2, ax=maxis2)

maxis1.set_xlabel('Number of lenders')
maxis2.set_xlabel('Number of lenders')

sns.despine()
plt.show()
df = kiva_loans[(kiva_loans['lender_count'] > 40) & (kiva_loans['loan_amount'] < 10000) & (kiva_loans['posted_to_funded_time_in_hours'] < 6000)]#.sample(1000)
xval = df.lender_count
yval = df.posted_to_funded_time_in_hours
cval = df.loan_amount

if PUBLISH:
    plt.scatter(x=xval.values, y=yval.values, c=cval.values, cmap=plt.get_cmap('jet'), alpha = 0.2)
    plt.title('Lender count vs. time from posting to funding, color = loan amount')
    plt.xlabel("Lender count")
    plt.ylabel("Time from posting to funding (hours)")
    cbar = plt.colorbar()
    cbar.set_label('Loan amount ($)', rotation=90)

    # save this one so it shows up in feed with this image as the output
    plt.savefig("fname2.png")
    plt.show()
kiva_loans['funding_per_lender'] = kiva_loans['loan_amount']/kiva_loans['lender_count']
kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'funding_per_lender'] = np.nan

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("All lender counts")
maxis2.set_title("Focus on lender counts < 100")

sns.kdeplot(kiva_loans.funding_per_lender, 
               color='#3CB371',shade=True, label='', bw=100, ax=maxis1)

sns.kdeplot(kiva_loans[(kiva_loans['funding_per_lender'] < 200)].funding_per_lender, 
               color='#3CB371',shade=True, label='', bw=5, ax=maxis2)

maxis1.set_xlabel('Funding per lender ($)')
maxis2.set_xlabel('Funding per lender ($)')

sns.despine()
plt.show()
fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("All loan terms")
maxis2.set_title("Loan terms up to 3 years")

sns.kdeplot(kiva_loans.term_in_months, 
               color='#3CB371',shade=True, label='', bw=0.5, ax=maxis1)

sns.kdeplot(kiva_loans[(kiva_loans['term_in_months'] <= 36)].term_in_months, 
               color='#3CB371',shade=True, label='', bw=0.5, ax=maxis2)

sns.kdeplot(kiva_loans[(kiva_loans['term_in_months'] <= 36) & (kiva_loans['percent_female'] == 1)].term_in_months, 
               color='#FF6347',shade=False, label='female', bw=0.5, ax=maxis2)
sns.kdeplot(kiva_loans[(kiva_loans['term_in_months'] <= 36) & (kiva_loans['percent_female'] == 0)].term_in_months, 
               color='#5DADE2',shade=False, label='male', bw=0.5, ax=maxis2)

maxis1.set_xlabel('Loan term (months)')
maxis2.set_xlabel('Loan term (months)')

maxis2.axvline(x=8, color = 'black', lw = 1) 
maxis2.axvline(x=11, color = 'black', lw = 1) 
maxis2.axvline(x=14, color = 'black', lw = 1) 

female = kiva_loans[(kiva_loans['percent_female'] == 1)].term_in_months
male   = kiva_loans[(kiva_loans['percent_female'] == 0)].term_in_months

print("Mean repayment for females {:0.2f}, median {:0.2f}, mode {:0.2f}". format(female.mean(), female.median(), female.mode()[0]))
print("Mean repayment for males {:0.2f}, median {:0.2f}, mode {:0.2f}". format(male.mean(), male.median(), male.mode()[0]))

sns.despine()
df = kiva_loans[(kiva_loans['loan_amount'] < 10000)]
xval = df.term_in_months
yval = df.loan_amount
cval = df.percent_female*100

if PUBLISH:
    plt.scatter(x=xval.values, y=yval.values, c=cval.values, cmap=plt.get_cmap('jet'), alpha = 0.2)
    plt.title('Loan amount vs loan term, color = percent female')
    plt.xlabel("Loan term (months)")
    plt.ylabel("Loan amount ($)")
    cbar = plt.colorbar()
    cbar.set_label('Percent female (%)', rotation=90)
fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("Repayment intervals")
maxis2.set_title("Repayment intervals (log scale)")

intervalcounts = kiva_loans.repayment_interval.value_counts()

sns.barplot(x=intervalcounts.index, 
            y=intervalcounts, ax = maxis1, palette="Greens_d", edgecolor=['black', 'black', 'black', 'black'])

sns.barplot(x=intervalcounts.index, 
            y=intervalcounts, ax = maxis2, palette="Greens_d", edgecolor=['black', 'black', 'black', 'black'])

maxis1.set_ylabel("Frequency")
maxis2.set_ylabel("Frequency (log scale)")

maxis2.set_yscale('log', nonposy='clip')

sns.despine()
sns.violinplot(x="repayment_interval", y="loan_amount", hue="team_gender", data=kiva_loans[(kiva_loans['team_gender'] != 'mixed')&(kiva_loans['loan_amount'] <= 4000)], split=True,
               inner="quart", palette={"male": "#5DADE2", "female": "#FF6347"})
plt.xlabel('Repayment interval')
plt.xlabel('Loan amount')
plt.title('Repayment interval vs loan amount for gender')
plt.legend(title = 'Gender')

sns.despine()
plt.show()
fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("Female")
maxis2.set_title("Male")

intervalcounts_male = kiva_loans[kiva_loans['percent_female'] == 0].repayment_interval.value_counts()
intervalcounts_female = kiva_loans[kiva_loans['percent_female'] == 1].repayment_interval.value_counts()

sns.barplot(x=intervalcounts_female.index, 
            y=intervalcounts_female, ax = maxis1, palette="Reds_d", edgecolor=['black', 'black', 'black', 'black'])

sns.barplot(x=intervalcounts_male.index, 
            y=intervalcounts_male, ax = maxis2, palette="Blues_d", edgecolor=['black', 'black', 'black', 'black'])

sns.despine()
plt.show()
kiva_loans[['sector', 'activity', 'use', 'tags']].describe()
wordcloud = WordCloud(background_color='white', width = 1400, height = 600, max_words = 15, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(kiva_loans['sector'].value_counts()))
plt.axis('off')
plt.title('Word cloud of sector')
plt.savefig("fname1.png")
plt.show()
wordcloud = WordCloud(background_color='white', width = 1400, height = 600, max_words = 163, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(kiva_loans['activity'].value_counts()))
plt.axis('off')
plt.title('Word cloud of loan activity')
plt.show()
kiva_loans['use_simplified'] = kiva_loans['use'].copy()

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('clean water').fillna(False), 'use_simplified'] = 'clean water'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('water filter').fillna(False), 'use_simplified'] = 'clean water'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('water filtration').fillna(False), 'use_simplified'] = 'clean water'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('toilet').fillna(False), 'use_simplified'] = 'sanitary toilet'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('latrine').fillna(False), 'use_simplified'] = 'sanitary toilet'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('university').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('school').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('tuition').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('studies').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('study').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('college').fillna(False), 'use_simplified'] = 'school'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('supplies to raise').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('feed and vitamins').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('feeds and vitamins').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('fertilizer').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('farm').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('maize').fillna(False), 'use_simplified'] = 'farm supplies'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' to sell').fillna(False), 'use_simplified'] = 'merchandice to sell'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' for resale').fillna(False), 'use_simplified'] = 'merchandice to sell'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' more stock').fillna(False), 'use_simplified'] = 'merchandice to sell'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' restock').fillna(False), 'use_simplified'] = 'merchandice to sell'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('food production business').fillna(False), 'use_simplified'] = 'food production business'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('solar lamp').fillna(False), 'use_simplified'] = 'solar lamp'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('solar lantern').fillna(False), 'use_simplified'] = 'solar lamp'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('solar light').fillna(False), 'use_simplified'] = 'solar lamp'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('building materials').fillna(False), 'use_simplified'] = 'building materials'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('construction materials').fillna(False), 'use_simplified'] = 'building materials'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('cement').fillna(False), 'use_simplified'] = 'building materials'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' fish ').fillna(False), 'use_simplified'] = 'fish'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' seafood ').fillna(False), 'use_simplified'] = 'fish'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' fishing ').fillna(False), 'use_simplified'] = 'fish'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('stove').fillna(False), 'use_simplified'] = 'stove'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('cattle').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('calves').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('cow').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('poultry').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('pig').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('goat').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('chicken').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('livestock').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('buffalo').fillna(False), 'use_simplified'] = 'livestock'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('items to sell').fillna(False), 'use_simplified'] = 'items to sell'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('clothes').fillna(False), 'use_simplified'] = 'clothes'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('clothing').fillna(False), 'use_simplified'] = 'clothes'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('tailor').fillna(False), 'use_simplified'] = 'clothes'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('sewing').fillna(False), 'use_simplified'] = 'clothes'

print("Using simple regular expression to simplify the use variable. After simplification the number of unique values is", 
      kiva_loans['use_simplified'].nunique(), "which is down from", kiva_loans['use'].nunique(),"in the original dataset")

wordcloud = WordCloud(background_color='white', width = 1400, height = 600, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(kiva_loans['use_simplified'].value_counts().head(200)))
plt.axis('off')
plt.title('Word cloud of simplified use')
plt.show()
s = kiva_loans['tags'].value_counts().head(300)
tupples = list(zip(s.index, s))
cleaned = pd.Series(dict([i for i in tupples if '#' in i[0]]))

wordcloud = WordCloud(background_color='white', width = 1400, height = 600, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(cleaned.head(200)))
plt.axis('off')
plt.title('Word cloud of loan tags (just hashtags)')
plt.show()

#load data
kiva_mpi_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")

# explore a little
print("Total entries in kiva_mpi_locations:", len(kiva_mpi_locations))
print("Dont trust the column metrics that they give you - there are null values hiding in here:\n", kiva_mpi_locations.isnull().sum())
print("The most frequent location is (lat,lng) ", kiva_mpi_locations['geo'].value_counts()[:1])

kiva_mpi_locations = kiva_mpi_locations.dropna(axis=0, how='any')
print(" *"* 30)
print("After dropna")
print(" *"* 30)

print("Total entries in kiva_mpi_locations:", len(kiva_mpi_locations))
print("Dont trust the column metrics that they give you - there are null values hiding in here:\n", kiva_mpi_locations.isnull().sum())
print("The most frequent location is (lat,lng) ", kiva_mpi_locations['geo'].value_counts()[:1])

#kiva_mpi_locations.sample(10)
kiva_mpi_locations[kiva_mpi_locations['ISO'] == 'USA']
fig = plt.figure(figsize=(15,6))

m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()

x, y = m(kiva_mpi_locations['lon'].values, kiva_mpi_locations['lat'].values)

m.scatter(x, y, latlon=True,
          c=kiva_mpi_locations['MPI'].values, #s=area,
          cmap='jet', alpha=0.5)
plt.colorbar()
plt.title('MPI map - higher # means more poverty')
plt.show()
fig = plt.figure(figsize=(15,6))

m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()

nepal = kiva_mpi_locations[kiva_mpi_locations['country'] == 'Nepal']
is_in_nepal = (nepal['lon'] > 80)
locs_in_nepal = is_in_nepal.sum()
locs_labeled_nepal = len(nepal)
is_in_nepal[is_in_nepal] = 'blue'
is_in_nepal[is_in_nepal==False] = 'red'

x, y = m(nepal['lon'].values, nepal['lat'].values)

m.scatter(x, y, latlon=True,
          c=is_in_nepal, #s=area,
          alpha=1)

plt.title('MPI map of Nepal labeled locations - only {} of {} are in Nepal'.format(locs_in_nepal, locs_labeled_nepal))
plt.savefig("fname0.png")
kiva_mpi_locations[kiva_mpi_locations['country'] == 'Nepal']