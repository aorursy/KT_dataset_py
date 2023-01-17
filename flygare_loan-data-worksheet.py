import numpy as np

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')

import seaborn as sns

import datetime as dt
df = pd.read_csv("../input/Loan payments data.csv")

for c in ['effective_date','paid_off_time','due_date']:

    df[c]=pd.to_datetime(df[c])

df.past_due_days.fillna(0, inplace=True)
df.info()
df.Loan_ID.nunique()
print (f"Unique values:", df.loan_status.unique()[:])

g=pd.DataFrame(df.groupby('loan_status')['Loan_ID'].count())

g
print (f"Unique values:", df.Principal.unique()[:])

g=pd.DataFrame(df.groupby('Principal')['Loan_ID'].count())

g
l=list(g.index)

l
print (f"Unique values:", df.terms.unique()[:])

mapTerms={7:'weekly', 15:'bi-weekly', 30: 'monthly'}

g=pd.DataFrame(df.groupby('terms')['Loan_ID'].count())

g
fig, ax = plt.subplots(figsize=(5,5))

pos = np.arange(len(df.terms.unique()))

ax.pie(g.Loan_ID.values, labels=[mapTerms[l] for l in list(g.index)])

ax.set_title('Loans (count) by contract term');
print (f"Unique values:", df['effective_date'].unique()[:])

g=pd.DataFrame(df.groupby('effective_date')['Loan_ID'].count())

g.loc[:,'Weekday']=pd.Series(g.index, index=g.index).dt.weekday_name

g.loc[:,'strDate']=pd.Series(g.index, index=g.index).dt.date

#g
fig, ax = plt.subplots(figsize=(5,5))

pos = np.arange(len(df.effective_date.unique()))

ax.bar(pos, height=g.Loan_ID.values, color=['b','b','g','g','b','b','b'])

ax.set_xticks(pos)

ax.set_xticklabels([w+"\n, "+ d.strftime('%d %b %Y') for w, d in zip(g.Weekday.values,g.strDate.values)],

                   rotation=90)

ax.set_title('Loans (count) by origination date');
print (f"Unique values:", df['due_date'].unique()[:])

g=pd.DataFrame(df.groupby('due_date')['Loan_ID'].count())

#g
df['TDterms']=df['terms'].astype('timedelta64[D]')

df['recalc_due_date']=df['effective_date']+df['TDterms']+dt.timedelta(days=-1)

mm=df[['effective_date', 'due_date', 'terms','recalc_due_date']][df['due_date']!=df['recalc_due_date']]

print('Mismatches in recalculated due_date: '+ str(len(mm)))

mm

#df[['effective_date', 'due_date', 'terms','recalc_due_date']].head()
print (f"Unique values:", df.paid_off_time.nunique())

#g=pd.DataFrame(df.groupby('paid_off_time')['Loan_ID'].count())

#g
print (f"Unique values:", df.past_due_days.unique()[:])

g=pd.DataFrame(df.groupby(['loan_status', 'past_due_days'])['Loan_ID'].count())

g=g.unstack('loan_status')

g=g.fillna(0)

#g
print(f'Collection min days: ',df[df.loan_status=='COLLECTION'].past_due_days.min())

print(f'Collection max days: ',df[df.loan_status=='COLLECTION'].past_due_days.max())

print(f'Paidoff min days: ',df[df.loan_status=='COLLECTION_PAIDOFF'].past_due_days.min())

print(f'Paidoff max days: ',df[df.loan_status=='COLLECTION_PAIDOFF'].past_due_days.max())
fig, ax = plt.subplots(figsize=(13.5,5))

pos = g.loc[:,'Loan_ID'].index

ax.bar(left=pos,

       height=g.loc[:,'Loan_ID']['COLLECTION'].values,

       label='Collection')

ax.bar(left=pos,

       bottom=g.loc[:,'Loan_ID']['COLLECTION'].values,

       height=g.loc[:,'Loan_ID']['COLLECTION_PAIDOFF'].values,

       label='Paid off')

ax.set_xticks([5*x for x in range(np.int(max(pos)/5.0)+1)])

ax.legend()

ax.set_title('Loans (count) by origination date');
print (f"Unique values:", df.age.unique()[:])

g=pd.DataFrame(df.groupby('age')['Loan_ID'].count())

#g
fig, ax = plt.subplots(figsize=(13.5,5))

pos = g.index

ax.bar(left=pos,

       height=g.loc[:,'Loan_ID'].values,

       label='Age')

ax.set_xticks([5*x for x in range(np.int(max(pos)/5.0)+1)])

ax.legend()

ax.set_title('Borrower Age (count)');
print (f"Unique values:", df.education.unique()[:])

g=pd.DataFrame(df.groupby('education')['Loan_ID'].count())

#g
fig, ax = plt.subplots(figsize=(5,5))

ax.pie(g.Loan_ID.values, labels=list(g.index))

ax.set_title('Education (count of borrows)');
print (f"Unique values:", df.Gender.unique()[:])

g=pd.DataFrame(df.groupby('Gender')['Loan_ID'].count())

#g
fig, ax = plt.subplots(figsize=(5,5))

ax.pie(g.Loan_ID.values, labels=list(g.index))

ax.set_title('Gemder (count of borrows)');
mapGender0 = {'male':0, 'female': 1}

df['Gender0']=df['Gender'].map(mapGender0)

mapEducation0 = {'High School or Below': 0, 'Bechalor': 1, 'college' : 2 , 'Master or Above':3}

df['Education0']=df['education'].map(mapEducation0)
corr = df[['age','Gender0','Education0']].corr()

fig, ax = plt.subplots(figsize = (6, 5))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

map   = sns.heatmap(

        corr, 

        cmap = plt.cm.coolwarm,

        square=True, 

        cbar_kws={'shrink': .9}, 

        ax=ax, 

        annot = True, 

        annot_kws={'fontsize': 12})