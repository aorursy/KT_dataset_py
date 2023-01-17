import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.tools.plotting
import seaborn as sns
import matplotlib
import squarify
%matplotlib inline

plt.style.use('seaborn')
codebook = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Codebook.csv')
numeric_mapping = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv')
numeric = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric.csv', na_values=['#NULL!', 'nan'])
values = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv', na_values=['#NULL!', 'nan'])

codebook.head()
numeric_mapping.head()
numeric.head()
values.head()
codebook.columns = ['fieldname', 'question', 'notes']
codebook.set_index('fieldname', inplace=True);
numeric_mapping.set_index('Data Field', inplace=True)

numeric.q1AgeBeginCoding = numeric.q1AgeBeginCoding.astype(float)
numeric.q2Age = numeric.q2Age.astype(float)
numeric = numeric.fillna(-1)

values = values.fillna('Not provided')
print(values.columns.ravel())
def draw_heatmap(column1, column2, title=None, annot=True, ax=None, size=(10, 10), data=values):
    cross = pd.crosstab(data[column1], data[column2])
    
    if ax is None:
        f, ax = plt.subplots(figsize=size)
        
    sns.heatmap(cross, cmap='Reds', annot=annot, ax=ax)
    ax.set_ylabel(codebook.loc[column1]['question'])
    ax.set_xlabel(codebook.loc[column2]['question'])
    
    if title is not None:
        ax.set_title(title)
# We need to shift NaN to 0, because data starts from value 1
numeric.loc[numeric['q1AgeBeginCoding'] == -1, 'q1AgeBeginCoding'] = 0
numeric.loc[numeric['q2Age'] == -1, 'q2Age'] = 0

# And to trim text so that it fits plots
numeric_mapping.loc['q2Age'] = [[i+1, j] for i, j in zip(range(9), ['Under 12', '12 - 18', '18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 - 74', '75+'])]
numeric_mapping.loc['q1AgeBeginCoding'] = numeric_mapping.loc['q1AgeBeginCoding'].applymap(lambda x: str(x).replace('years old', ''))
# I will be frequently using semicolons to suppress matplotlib output
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))

sns.set(font_scale=1)
count = sns.countplot(x='q3Gender', data=numeric, ax=ax[0][0])
count.set_xticklabels(np.append(['Not provided'], numeric_mapping.loc['q3Gender'].values[:, 1]));
count.set_xlabel('Gender')

bar = sns.barplot(x='q2Age', y='q1AgeBeginCoding', hue='q3Gender', data=numeric, ax=ax[0][1])
ax[0][1].yaxis.set_ticks([i for i in range(len(numeric_mapping.loc['q1AgeBeginCoding'].values[:, 1]))])
bar.set_yticklabels(np.append(['Not provided'], numeric_mapping.loc['q1AgeBeginCoding'].values[:, 1]))
bar.set_xticklabels(np.append(['Not provided'], numeric_mapping.loc['q2Age'].values[:, 1]));
bar.set_xlabel(codebook.loc['q2Age']['question'])
bar.set_ylabel(codebook.loc['q1AgeBeginCoding']['question'])

bar.legend(loc=2)
for i, j in zip(bar.get_legend().texts, np.append(['Not provided'], numeric_mapping.loc['q3Gender'].values[:, 1])):
    i.set_text(j)
bar.get_legend().set_title('')

# fig.tight_layout()
sns.set()
draw_heatmap('q1AgeBeginCoding', 'q2Age', ax=ax[1][0], annot=False)
draw_heatmap('q2Age', 'q3Gender', ax=ax[1][1], annot=True, size=(5, 5))
fig.tight_layout()
plt.savefig('basic_info.jpg')
trimmed_numeric = numeric.loc[(numeric['q2Age']<6) & (numeric['q2Age']>1)]

sns.set()
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 5));

sns.factorplot(x='q2Age', y='q1AgeBeginCoding', data=trimmed_numeric, ax=ax[0], color='y')
sns.barplot(x='q2Age', y='q1AgeBeginCoding', hue='q3Gender', data=trimmed_numeric, ax=ax[0], alpha=1);
ax[0].set_yticks(range(6))
ax[0].set_yticklabels(np.append(['Not provided'], numeric_mapping.loc['q1AgeBeginCoding'].values[:4, 1]))
ax[0].set_ylabel(codebook.loc['q1AgeBeginCoding']['question'])
ax[0].set_xlabel(codebook.loc['q2Age']['question'])
ax[0].set_xticklabels(numeric_mapping.loc['q2Age'].values[1:, 1])
ax[0].get_legend().set_title('')
# This could have been done better, but I was unable to figure this out
ax[0].set_title('Average age for all genders (bars) and collective (line)')

sns.countplot(trimmed_numeric['q2Age'], ax=ax[1])
ax[1].set_title('Age distribution')
ax[1].set_xlabel('')
ax[1].set_xlabel(codebook.loc['q2Age']['question'])
ax[1].set_xticklabels(numeric_mapping.loc['q2Age'].values[1:, 1])

for i, j in zip(ax[0].get_legend().texts, np.append(['Not provided'], numeric_mapping.loc['q3Gender'].values[:, 1])):
    i.set_text(j)

# A small hack to get around seaborn generating unneccessary plots
plt.clf();

f = plt.figure(figsize=(8,5));
factAx = plt.gca();
sns.factorplot(x='q2Age', y='q1AgeBeginCoding', hue='q3Gender', data=trimmed_numeric, legend=False, kind='violin', ax=factAx);
factAx.yaxis.set_ticks([i for i in range(len(numeric_mapping.loc['q1AgeBeginCoding'].values[:, 1]))])
factAx.set_xlabel(codebook.loc['q2Age']['question'])
factAx.set_xticklabels(numeric_mapping.loc['q2Age'].values[1:, 1])
factAx.set_yticklabels(np.append(['Not provided'], numeric_mapping.loc['q1AgeBeginCoding'].values[:, 1]))
factAx.set_ylabel(codebook.loc['q1AgeBeginCoding']['question'])
# ax.get_legend().remove()

for i, j in zip(factAx.get_legend().texts,np.append(['Not provided'], numeric_mapping.loc['q3Gender'].values[:, 1])):
    i.set_text(j)
factAx.get_legend().set_title('')

f.tight_layout();
plt.clf();
plt.savefig('age.jpg')
# Information about experience is stored in 5 columns (kind of like one hot encoding), so we need to reverse it
# We are going to follow three-table pattern in order to be able to use seaborn (it needs numeric data) 
columns = ['q6LearnCodeDontKnowHowToYet', 'q6LearnCodeOther',
           'q6LearnCodeAccelTrain', 'q6LearnCodeSelfTaught', 'q6LearnCodeUni']

res = np.where(values[columns[0]]!='Not provided', 0, -1)
res_val = np.where(values[columns[0]]!='Not provided', "Didn't", 'Not provided')
for i, j in enumerate(columns[1:]):
    res[values[j]!='Not provided'] = i
    res_val[values[j]!='Not provided'] = j.split('LearnCode')[-1]
    
numeric['q6LearnCode'] = res
values['q6LearnCode'] = res_val

numeric_mapping = numeric_mapping.append(pd.DataFrame(
    {'Data Field': 'q6LearnCode',
     'Value': [i-1 for i in range(6)], 
     'Label': ['Not provided', "Didn't", 'Other way', 'Self taught', 'Accel Train', 'University']}
    ).set_index('Data Field')) 

codebook.loc['q6LearnCode'] = 'How did you learn how to code?'
# Clearing fonts settings
sns.set()
plt.figure(figsize=(16,5))
ax = plt.subplot(121)
draw_heatmap('q2Age', 'q6LearnCode', title='Learning means by age', annot=False, ax=ax)
ax = plt.subplot(122)
draw_heatmap('q1AgeBeginCoding', 'q6LearnCode', title='Learning means by age of starting coding', annot=True, ax=ax)
plt.tight_layout()
plt.savefig('learning_means.jpg')
# First we will have to do a little preprocessing of columns

columns = ['q25LangC', 'q25LangCPlusPlus', 'q25LangJava', 'q25LangPython',
         'q25LangRuby', 'q25LangJavascript', 'q25LangCSharp', 'q25LangGo', 'q25Scala',
         'q25LangPerl', 'q25LangSwift', 'q25LangPascal', 'q25LangClojure', 'q25LangPHP',
         'q25LangHaskell', 'q25LangLua', 'q25LangR', 'q25LangRust', 'q25LangTypescript',
         'q25LangKotlin', 'q25LangJulia', 'q25LangErlang', 'q25LangOcaml']

res = np.where(values[columns[0]]!='Not provided', 0, 1)

for i in columns:
    numeric[i+'WillLearn'] = np.where(values[i]=='Will Learn', 1, 0)
    numeric[i+'Know'] = np.where(values[i]=='Know', 1, 0)
plt.figure(figsize=(8, 5))
  
for i, j in enumerate(columns):
    plt.barh(i, np.sum(numeric[j+'Know']) + np.sum(numeric[j+'WillLearn']), color='orange')

for i, j in enumerate(columns):
    plt.barh(i, np.sum(numeric[j+'Know']), color='#005aff')

plt.gca().set_yticks(range(len(columns)));
plt.gca().set_yticklabels([j.split('Lang')[-1] for j in columns]);
plt.title('Languages popularity on HackerRank');

custom_lines = [matplotlib.patches.Patch(color='#005aff', lw=1),
                matplotlib.patches.Patch(color='orange', lw=1)]
    
plt.legend(custom_lines, ['Know language', 'Want to learn language']);
plt.gca().get_legend().set_title('Number of developers that')

plt.tight_layout()

plt.savefig('language_popularity.jpg')
plt.figure(figsize=(16, 10))

res_will_learn = np.array(np.sum(numeric[numeric['q2Age']==0][[j+'WillLearn' for j in columns]]).values)/len(numeric[numeric['q2Age']==0])
res_know = np.array(np.sum(numeric[numeric['q2Age']==0][[j+'Know' for j in columns]]).values)/len(numeric[numeric['q2Age']==0])
res_everything = np.array((np.sum(numeric[numeric['q2Age']==0][[j+'Know' for j in columns]]).values \
               + np.sum(numeric[numeric['q2Age']==0][[j+'WillLearn' for j in columns]]).values) \
               / len(numeric[numeric['q2Age']==0]))
for i in list(set(numeric['q2Age']))[1:]:
    res_will_learn = np.vstack((res_will_learn, 
                                np.sum(numeric[numeric['q2Age']==i][[j+'WillLearn' for j in columns]]).values \
                                /len(numeric[numeric['q2Age']==i])))
    res_know = np.vstack((res_know, 
                          np.sum(numeric[numeric['q2Age']==i][[j+'Know' for j in columns]]).values \
                          /len(numeric[numeric['q2Age']==i])))
    res_everything = np.vstack((res_everything, 
                          (np.sum(numeric[numeric['q2Age']==i][[j+'Know' for j in columns]]).values \
                        + np.sum(numeric[numeric['q2Age']==i][[j+'WillLearn' for j in columns]]).values) \
                        / len(numeric[numeric['q2Age']==i])))
    
ax1 = plt.subplot(221)
sns.heatmap(res_will_learn, ax=ax1);
ax1.set_yticklabels(np.append(['Not provided'], numeric_mapping.loc['q2Age'].values[:, 0]), rotation='horizontal')
ax1.set_xticklabels([j.split('Lang')[-1] for j in columns], rotation='vertical');
ax1.set_title('Percentage of developers that want to learn languages in each age category ');

ax2 = plt.subplot(222)
sns.heatmap(res_know, ax=ax2);
ax2.set_yticklabels(np.append(['Not provided'], numeric_mapping.loc['q2Age'].values[:, 0]), rotation='horizontal')
ax2.set_xticklabels([j.split('Lang')[-1] for j in columns], rotation='vertical');
ax2.set_title('Percentage of known languages in each age category ');

ax3 = plt.subplot(223)
sns.heatmap(res_everything, ax=ax3);
ax3.set_yticklabels(np.append(['Not provided'], numeric_mapping.loc['q2Age'].values[:, 0]), rotation='horizontal')
ax3.set_xticklabels([j.split('Lang')[-1] for j in columns], rotation='vertical');
ax3.set_title('Percentage of developers that know or want to learn language')
plt.tight_layout()
plt.savefig('languages_heatmap.jpg')
columns = [i for i in values.columns.ravel() if 'q28' in i]
langs_known = [i for i in values.columns.ravel() if 'q25' in i]
columns = columns[:-1]
langs_known = langs_known[:-1]

plt.figure(figsize=(16,5))

plt.subplot(121)
love_height = []
hate_height = []

for i, j in enumerate(zip(columns, langs_known)):
    love = len(numeric[(numeric[j[1]]>=1) & (values[j[0]]=='Love')])/(len(numeric[numeric[j[1]]>=1]))
    plt.bar(i, love, color='#4c72b0')
    plt.text(i, love-0.05, '%i' % int(love*100), horizontalalignment='center', size=10, color='white')
    
    hate = len(numeric[(numeric[j[1]]>=1) & (values[j[0]]=='Hate')])/(len(numeric[numeric[j[1]]>=1]))
    plt.bar(i, -hate, color='#55a868')
    plt.text(i, -hate+0.01, '%i' % int(hate*100), horizontalalignment='center', size=10, color='white')
    
    love_height.append(love)
    hate_height.append(hate)
    
    
custom_lines = [matplotlib.patches.Patch(color='#4c72b0', lw=1),
                matplotlib.patches.Patch(color='#55a868', lw=1),
                matplotlib.lines.Line2D([0], [0], color='orange')]
    
plt.legend(custom_lines, ['Love', 'Hate', 'Overall reputation'])
plt.plot([(i-j)/2 for i, j in zip(love_height, hate_height)], color='orange')

plt.gca().set_xticks(range(len(columns)))
plt.gca().set_xticklabels([j.split('Love')[-1] for j in columns], rotation='vertical');
plt.gca().set_title('Reputation of languages that developers know or will learn');
plt.gca().set_yticklabels(['%i%%' % abs(i*100) for i in plt.yticks()[0]]);
plt.ylabel('Percentage of users');

plt.subplot(122)
love_height = []
hate_height = []

for i, j in enumerate(zip(columns, langs_known)):
    love = len(numeric[(numeric[j[1]]==0) & (values[j[0]]=='Love')])/(len(numeric[numeric[j[1]]>=1]))
    plt.bar(i, love, color='#4c72b0')
    plt.text(i, love-0.02, '%i' % int(love*100), horizontalalignment='center', size=10, color='white')
    
    hate = len(numeric[(numeric[j[1]]==0) & (values[j[0]]=='Hate')])/(len(numeric[numeric[j[1]]>=1]))
    plt.bar(i, -hate, color='#55a868')
    plt.text(i, -hate+0.01, '%i' % int(hate*100), horizontalalignment='center', size=10, color='white')
    
    love_height.append(love)
    hate_height.append(hate)
    
    
plt.plot([(i-j)/2 for i, j in zip(love_height, hate_height)], color='orange')

plt.gca().set_xticks(range(len(columns)))
plt.gca().set_xticklabels([j.split('Love')[-1] for j in columns], rotation='vertical');
plt.gca().set_title('Reputation of languages that developers did not nor will not learn');
plt.gca().set_yticklabels(['%i%%' % abs(i*100) for i in plt.yticks()[0]]);
plt.ylabel('Percentage of users');
plt.savefig('languages_reputation.jpg')
plt.figure(figsize=(16,5))

plt.subplot(121)
love_height = []
hate_height = []

for i, j in enumerate(zip(columns, langs_known)):
    love = len(numeric[(values[j[1]]=='Know') & (values[j[0]]=='Love')])/(len(numeric[values[j[1]]=='Know']))
    plt.bar(i, love, color='#4c72b0')
    plt.text(i, love-0.05, '%i' % int(love*100), horizontalalignment='center', size=10, color='white')
    
    hate = len(numeric[(values[j[1]]=='Know') & (values[j[0]]=='Hate')])/(len(numeric[values[j[1]]=='Know']))
    plt.bar(i, -hate, color='#55a868')
    plt.text(i, -hate+0.01, '%i' % int(hate*100), horizontalalignment='center', size=10, color='white')
    
    love_height.append(love)
    hate_height.append(hate)
    
    
custom_lines = [matplotlib.patches.Patch(color='#4c72b0', lw=1),
                matplotlib.patches.Patch(color='#55a868', lw=1),
                matplotlib.lines.Line2D([0], [0], color='orange')]
    
plt.legend(custom_lines, ['Love', 'Hate', 'Overall reputation'])
plt.plot([(i-j)/2 for i, j in zip(love_height, hate_height)], color='orange')

plt.gca().set_xticks(range(len(columns)))
plt.gca().set_xticklabels([j.split('Love')[-1] for j in columns], rotation='vertical');
plt.gca().set_title('Reputation of languages that developers already know');
plt.gca().set_yticklabels(['%i%%' % abs(i*100) for i in plt.yticks()[0]]);
plt.ylabel('Percentage of users');

plt.subplot(122)
love_height = []
hate_height = []

for i, j in enumerate(zip(columns, langs_known)):
    love = len(numeric[(values[j[1]]=='Will Learn') & (values[j[0]]=='Love')])/(len(numeric[values[j[1]]=='Will Learn']))
    plt.bar(i, love, color='#4c72b0')
    plt.text(i, love-0.05, '%i' % int(love*100), horizontalalignment='center', size=10, color='white')
    
    hate = len(numeric[(values[j[1]]=='Will Learn') & (values[j[0]]=='Hate')])/(len(numeric[values[j[1]]=='Will Learn']))
    plt.bar(i, -hate, color='#55a868')
    plt.text(i, -hate+0.01, '%i' % int(hate*100), horizontalalignment='center', size=10, color='white')
    
    love_height.append(love)
    hate_height.append(hate)
    
plt.plot([(i-j)/2 for i, j in zip(love_height, hate_height)], color='orange')

plt.gca().set_xticks(range(len(columns)))
plt.gca().set_xticklabels([j.split('Love')[-1] for j in columns], rotation='vertical');
plt.gca().set_title('Reputation of languages that developers are going to learn');
plt.gca().set_yticklabels(['%i%%' % abs(i*100) for i in plt.yticks()[0]]);
plt.ylabel('Percentage of users');
plt.savefig('languages_opinion.jpg')
plt.figure(figsize=(20, 10))
columns = [i for i in values.columns.ravel() if 'q28' in i]
langs_prof = [i for i in values.columns.ravel() if 'q22' in i]
langs_prof = langs_prof[1:-1]
columns = columns[:len(langs_prof)]

langs_sum = len(numeric[values['q16HiringManager']=='Yes'])
langs_name = ['%s: %.1f%%' % (i.split('Prof')[-1], (len(numeric[numeric[i]==1])/langs_sum)*100) for i in langs_prof]
langs_count = [len(numeric[numeric[i]==1]) for i in langs_prof]

squarify.plot(sizes=langs_count, label=langs_name, alpha=0.7, color=list(np.random.rand(17,3)))
plt.axis('off')
plt.title(codebook.loc[langs_prof[0]][0]);
plt.savefig('languages_desired.jpg')