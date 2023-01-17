import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import ttest_ind

%matplotlib inline
plt.rcParams['figure.figsize'] = [20.0, 7.0]

plt.rcParams.update({'font.size': 22})



sns.set_palette('bright')

sns.set_style('whitegrid')

sns.set_context('talk')
info_df = pd.read_csv('../input/heroes_information.csv')

powers_df = pd.read_csv('../input/super_hero_powers.csv')
print(info_df.shape)

info_df.head()
info_df.columns
#drop unnamed column

info_df = info_df.drop(['Unnamed: 0'], axis=1)
#check for null values

info_df.isnull().sum()
#show rows where publisher is null

info_df[info_df.Publisher.isnull()]
# get rid of NaN values for publisher

info_df.Publisher.fillna('Other', inplace=True)
# -99 is not a valid height or weight so set to NaN

info_df.replace(-99.0, np.nan, inplace=True)
#update genders if -

info_df.Gender.replace('-', 'Non-Binary', inplace=True)
#update alignment if -

info_df.Alignment.replace('-', 'None', inplace=True)
info_df.head()
#plot number of superheros by gender

sns.countplot(x='Gender', data=info_df);



plt.title('Superheros by Gender')

plt.ylabel('Number of Superheros')

plt.show();
#create new dataframe with only height and weight values that are not null

info_2 = info_df.dropna(subset=['Height', 'Weight'], how='any')

info_2.isnull().sum()
#keep only columns of interest for plotting

info_2 = info_2[['name', 'Gender', 'Height', 'Weight', 'Alignment']]
#plot height and weight by gender

sns.pairplot(data=info_2, hue='Gender', size=5)



plt.title('Superhero Heights and Weights by Gender')

plt.show();
sns.distplot(info_2.Weight, kde=False)

plt.title('Histogram of Superhero Weight')

plt.show();
#another height and weight visualization

sns.kdeplot(info_2['Weight'], info_2['Height'])



plt.title('Superheros Height and Weight')

plt.show();
#plot alignment by gender

sns.countplot(info_2.Gender, hue=info_2.Alignment)



plt.title('Alignment by Gender')

plt.xlabel('Gender')

plt.ylabel('Number of Heros')

plt.show();
#neutral and none groups are both very small...combining into one group as neutral

info_2.Alignment.replace('None', 'neutral', inplace=True)

info_df.Alignment.replace('None', 'neutral', inplace=True)
info_2.Gender.value_counts()
info_2.Alignment.value_counts()
fig, ax = plt.subplots()

fig.subplots_adjust(hspace=0.25, wspace=0.25)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)



fig.add_subplot(221)

sns.swarmplot(x=info_2.Alignment, y=info_2.Weight, hue=info_2.Gender)



fig.add_subplot(222)

sns.stripplot(x=info_2.Alignment, y=info_2.Weight, hue=info_2.Gender, jitter=True)



fig.add_subplot(223)

sns.swarmplot(x=info_2.Gender, y=info_2.Weight, hue=info_2.Alignment)



fig.add_subplot(224)

sns.stripplot(x=info_2.Gender, y=info_2.Weight, hue=info_2.Alignment, jitter=True)



plt.title('Superhero Characteristics')

plt.show;
#plot number of superheros per publisher

sns.countplot(x="Publisher", data=info_df)



plt.title('Number of Superheros by Publisher')

plt.ylabel('Number of Superheros')

plt.xlabel('Publisher')

plt.xticks(rotation = 90)

plt.show();
#create new dataframe with only small publishers with < 200 superheros

counts = info_df.Publisher.value_counts()

small_publishers_df = info_df.loc[info_df['Publisher'].isin(counts[counts < 200].index), :]



small_publishers_count = small_publishers_df.Publisher.value_counts()
#plot only the small publishers

sns.countplot(x="Publisher", data=small_publishers_df)



plt.title('Number of Superheros by Publisher')

plt.ylabel('Number of Superheros')

plt.xlabel('Publisher')

plt.xticks(rotation = 90)

plt.show();
small_publishers_df.sort_values('Publisher')
powers_df.head(10)
# check if any NaN values

powers_df[powers_df.isnull().any(axis=1)]
#create new column with sum of each superheros powers

powers_df['sum_powers'] = powers_df.iloc[2:].sum(axis=1)
#sort by most powers

most_powers = powers_df.sort_values('sum_powers', ascending=False)

most_powers.head()
#plot superheros by number of powers

sns.countplot(x='sum_powers', data=powers_df)



plt.xlabel('Number of Powers')

plt.ylabel('Number of Superheros')

plt.title('Superheros by Number of Powers')

plt.show();
#plot superheros with the most powers

most_powers = powers_df[['hero_names', 'sum_powers']].sort_values('sum_powers', ascending=False)



sns.barplot(x=most_powers['hero_names'].head(20),

            y='sum_powers', 

            data=most_powers)



plt.xticks(rotation=80)

plt.title('Superheros with the Most Powers')

plt.xlabel('Superhero')

plt.ylabel('Number of Powers')

plt.show();
#create a word cloud of superheros with the most powers

from wordcloud import WordCloud



#remove spaces in between name parts - wordcloud splits on whitespace

heronames = most_powers['hero_names'].str.replace(' ', '')



wordcloud = WordCloud().generate(str(heronames.head(20)))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
#create new df with number of superheros with each power

sum_row = {col: powers_df[col].sum() for col in powers_df}



sum_df = pd.DataFrame(sum_row, index=["Total"])



sum_df = sum_df.drop('hero_names', axis=1)

sum_df = sum_df.drop('sum_powers', axis=1)





sum_df.tail()
#transform df

sum_df = sum_df.T

sum_df.head()
sum_df = sum_df.reset_index()

sum_df.rename(columns = {'index':'Powers'}, inplace=True)
sum_df.columns
#sort powers by most common

sum_df = sum_df.sort_values('Total', ascending=False)

sum_df.head()
#plot most common superpowers

sns.barplot(x=sum_df['Powers'].head(20),

            y='Total', 

            data=sum_df)



plt.xticks(rotation=80)

plt.title('Most Common Superpowers')

plt.xlabel('Superpower')

plt.ylabel('Superheros with Power')

plt.show();
#plot least common superpowers

sns.barplot(x=sum_df['Powers'].tail(20),

            y='Total', 

            data=sum_df)



plt.xticks(rotation=80)

plt.title('Least Common Superpowers')

plt.xlabel('Superpower')

plt.ylabel('Superheros with Power')

plt.show();
#merge original dataframes together

merged_df = pd.merge(info_df, powers_df, left_on='name', right_on='hero_names', how='inner')
#view new df head

merged_df.head()
merged_df.shape
#plot number of superpowers by publisher

sns.barplot(x='Publisher', y='sum_powers', data=merged_df)

plt.xticks(rotation = 90);



plt.title('Number of Powers per Superhero by Publisher')

plt.xlabel('Publisher')

plt.ylabel('Number of Powers')

plt.show();
#What superhero is from South Park?

merged_df.loc[merged_df['Publisher'] == 'South Park']
#Is he really a superhero?? where is Mario then!?

merged_df.loc[merged_df['Publisher'] == 'Microsoft']
merged_df.Publisher.value_counts()
#the small publishers have significantly less superheros so keeping only the superheros from Marvel and DC

merged_df = merged_df[(merged_df['Publisher'] == 'Marvel Comics') | (merged_df['Publisher'] == 'DC Comics')]
merged_df.Publisher.value_counts()
#t-test comparing number of male heroes to female

print(ttest_ind(merged_df['Gender'] == 'Male', merged_df['Gender'] == 'Female'))
#t-test comparing number of powers for males and females

ttest_ind(merged_df.dropna()['sum_powers'][merged_df['Gender'] == 'Male'],

          merged_df.dropna()['sum_powers'][merged_df['Gender'] =='Female'])
#find mean number of powers by gender

merged_df.groupby('Gender')['sum_powers'].mean()
#plot number of superpowers by publisher

sns.barplot(x='Publisher', y='sum_powers', hue='Gender', data=merged_df)



plt.title('Number of Powers per Superhero by Publisher')

plt.xlabel('Publisher')

plt.ylabel('Number of Powers')

plt.show();
#plot number of powers by Alignment

sns.boxplot(x=merged_df.Alignment, y=merged_df.sum_powers, hue=merged_df.Gender)



plt.title('Number of Powers per Superhero by Alignment')

plt.xlabel('Alignment')

plt.ylabel('Number of Powers')

plt.show();