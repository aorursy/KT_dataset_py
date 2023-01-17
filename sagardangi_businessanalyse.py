import pandas as pd

df = pd.read_csv("../input/Business Analyst Assignment  - Dataset.csv")

df.head()
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
df['Annual Income'].fillna(0, inplace=True)
keep = ['More than 5 to 8 Lacs', 

        'More than 8 to 12 lacs',

        "Family's annual income: 10 lakhs", 

        "Less than 3 Lacs",

       'More than 12 Lacs',

        'More than 3 to 5 Lacs',

        'Stipend less than 1.5 lacs',

       ]

mp = {}

for i in list(set(df['Annual Income'])):

    if i in keep:

        mp[i] = i

    else:

        mp[i] = '0 income'



df['Annual Income1'] = df['Annual Income'].map(mp)
d = {'0 income':0,

    "Family's annual income: 10 lakhs":1000000,

     'Less than 3 Lacs':150000,

     'More than 12 Lacs':1200000,

     'More than 3 to 5 Lacs':400000,

     'More than 5 to 8 Lacs': 650000,

     'More than 8 to 12 lacs':1000000,

     'Stipend less than 1.5 lacs':150000

    }



df['in'] = df['Annual Income1'].map(d)
d1 = {'Less than 5000':2500,

      'More than 10000 to 15000':12500,

      'More than 5000 to 8000':6500,

      'More than 8000 to 10000':9000

     }



d2 = {'Less than 3000':1500,

      'More than 3000 to 5000':4000,

      'More than 5000 to 8000':6000,

      'More than 8000 to 10000':9000,

}

df['c1'] = df['How much you can spend monthly for a rented house'].map(d1)

df['c2'] = df["How much you can spend monthly for services and amenities (food, water, electricity etc.)"].map(d2)

df['cost'] = df['c1'] + df['c2']
t = df[df['in']!=0]
t['%spend'] = (t['c2'] / t['in'])*100
t.groupby(['Age Group'])['%spend'].mean()
t.groupby(['Annual Income1'])['%spend'].mean()
df.groupby(['Age Group'])['How much you can spend monthly for a rented house'].value_counts()
# Age 18-23

(6500*34 + 2500*24 + 9000*13 + 12500*5)/(34+24+13+5)
# Age 24-29

(6500*14 + 9000*14 + 12500*7 + 2500*6 + 17500*1)/(14+14+7+6+1)
# Age 18

9000
df.groupby(['Annual Income1'])['How much you can spend monthly for a rented house'].value_counts()
# 0

(2500*12 + 6500*12 + 12500*4 + 9000*4)/(12 +12+4+4)
# 10

2500
# < 3

(6500*16 + 2500*14 + 9000*4)/(16+14+4)
# > 12

(9000*3 + 2500*1)/4
# 3-5

(6500*6 + 9000*5 + 12500*4 + 2500*1)/(16)
# 5-8

(9000*12 + 6500*10 + 12500*2)/(24)
# 8-12

(6500*3 + 12500*2 + 2500 + 17500*1)/7
# stipend

6500
['Rate on the scale of  satisfaction level (1 to 10) to the factors given below based on your past coliving or rental experience  [Food]',

'Rate on the scale of  satisfaction level (1 to 10) to the factors given below based on your past coliving or rental experience  [Hygiene  (Common areas plus Room)]',

'Rate on the scale of  satisfaction level (1 to 10) to the factors given below based on your past coliving or rental experience  [Security]',

'Rate on the scale of  satisfaction level (1 to 10) to the factors given below based on your past coliving or rental experience  [Maintenance]',

'Rate on the scale of  satisfaction level (1 to 10) to the factors given below based on your past coliving or rental experience  [Rental amount/ Security Deposit]',]
set(df['How much you can spend monthly for a rented house'])
set(df['How much you can spend monthly for services and amenities (food, water, electricity etc.)'])
def f(df, a, b):

    w = df[((df['How much you can spend monthly for a rented house']==a) )&

      ((df['How much you can spend monthly for services and amenities (food, water, electricity etc.)']==b)

      )]

    print(w['Age Group'].value_counts())

    print(w['Annual Income1'].value_counts())

    # print(w.groupby(['Age Group', 'Annual Income1'])['w'].value_counts())
f(df,'More than 8000 to 10000', 'More than 5000 to 8000')
f(df,'More than 10000 to 15000', 'More than 5000 to 8000')
f(df,'More than 8000 to 10000', 'More than 8000 to 10000')


f(df,'More than 10000 to 15000', 'More than 8000 to 10000')
w = df[((df['How much you can spend monthly for a rented house']=='More than 8000 to 10000') 

    |(df['How much you can spend monthly for a rented house']=='More than 10000 to 15000'))

       &

  ((df['How much you can spend monthly for services and amenities (food, water, electricity etc.)']=='More than 5000 to 8000')

      | (df['How much you can spend monthly for services and amenities (food, water, electricity etc.)']== 'More than 8000 to 10000'))

  ]
w['Age Group'].value_counts()
w['Annual Income1'].value_counts()
df.groupby(['Age Group'])['What kind of accommodation did you find '].value_counts()
set(df['Annual Income1'])
df.groupby(['Annual Income1'])['What kind of accommodation did you find '].value_counts()
df['Rate your past experience of finding accommodation on rent on the scale of difficulty level (1 to 5)'].mean()
df.groupby(['What kind of accommodation did you find '])['Rate your past experience of finding accommodation on rent on the scale of difficulty level (1 to 5)'].mean()
set(df['How long maximum have you stayed in a rental accommodation '])
df['How long maximum have you stayed in a rental accommodation '].value_counts()
df.groupby(['How long maximum have you stayed in a rental accommodation ']).std()
df['cat'] = df['Mark the appropriate  category for you'].map({'Others': 'Others', 

                                                             'Post Graduate Student':'students',

                                                             'Under Graduate Student':'students',

                                                             'Working Professional':"Working Professional"})

df.groupby(['cat']).mean()
set(df['How much you can spend monthly for services and amenities (food, water, electricity etc.)'])
df['cost'].describe()
df.groupby(['Age Group'])['cost'].describe()
# 24-29 and younger than 18
df.groupby(['Annual Income1'])['cost'].describe()
# more than 12, 3-5, 5-8, 8-12, stipend
df.groupby(['cat'])['cost'].describe()
# working