import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

import seaborn as sns
def get_font(url, file_path):

    import requests

    r = requests.get(url)

    print("Download: " + file_path)

    with open(file_path, "wb") as f:

        f.write(r.content)

        f.close()

    return file_path





url = "https://github.com/hufe09/pydata_practice/raw/master/fonts/msyh.ttf"

file_path = "msyh.ttf"
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全

pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

%matplotlib inline

font = FontProperties(fname=get_font(url, file_path),

                      size=12)  # 指定文字,用来正常显示中文标签



sns.set_style('darkgrid')
patients = pd.read_csv('../input/patients.csv')

treatments = pd.read_csv('../input/treatments.csv')

adverse_reactions = pd.read_csv('../input/adverse_reactions.csv')
#显示前5行

patients.head()
# 简单摘要

patients.info()
# 各列汇总统计数据

patients.describe()
# 随机样本

patients.sample(5)
#　检查缺失数据

patients[patients['address'].isnull()]
# 对surname列计数

patients.surname.value_counts()
# 对weight列值排序

patients.weight.sort_values()
# 打印address列重复值

patients[patients.address.duplicated()]
# 身高体重比例

weight_lbs =192.3

height_in = 72

703 * weight_lbs / (height_in*height_in)
treatments.head()
treatments.info()
treatments.describe()
sum(treatments.auralin.isnull())
sum(treatments.novodra.isnull())
adverse_reactions.head()
adverse_reactions.info()
adverse_reactions.describe()
all_colums = pd.Series(list(patients) + list(treatments) + list(adverse_reactions))

all_colums[all_colums.duplicated()]
patients_clean = patients.copy()

patients_clean.sample(3)
treatments_clean = treatments.copy()

treatments_clean.sample(3)
adverse_reactions_clean = adverse_reactions.copy()

adverse_reactions_clean.sample(3)
patients_clean['phone_number'] = patients_clean.contact.str.extract('((?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4})', expand=True)

patients_clean['phone_number'].head()
patients_clean['email'] = patients_clean.contact.str.extract('([a-zA-Z][a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z])', expand=True)

patients_clean['email'].sample(3)
patients_clean = patients_clean.drop('contact', axis=1)
patients_clean.sample(1)
treatments_clean = pd.melt(treatments_clean, 

                id_vars=['given_name', 'surname', 'hba1c_start', 'hba1c_end', 'hba1c_change'],

                var_name = 'treatment', value_name='does')

treatments_clean.head()
treatments_clean = treatments_clean[treatments_clean.does != '-']

treatments_clean.head()
treatments_clean['does_start'],treatments_clean['does_end'] = treatments_clean['does'].str.split(' - ', 1).str

treatments_clean.head()
treatments_clean =treatments_clean.drop('does', axis = 1)

treatments_clean.head()
treatments_clean.does_start = treatments_clean.does_start.str.strip('u').astype(int)

treatments_clean.does_end = treatments_clean.does_end.str.strip('u').astype(int)

treatments_clean.head()
treatments_clean["does_change"] = treatments_clean["does_start"] - treatments_clean["does_end"]
treatments_clean.head()
treatments_clean = pd.merge(treatments_clean, adverse_reactions_clean, 

                            on=['given_name', 'surname'], how='left')

treatments_clean.sample()
treatments_clean.adverse_reaction.value_counts()
id_names = patients_clean[['patient_id', 'given_name', 'surname']]

id_names['given_name'] = id_names['given_name'].str.lower()

id_names['surname'] = id_names['surname'].str.lower()

treatments_clean = pd.merge(treatments_clean, id_names, on=['given_name', 'surname'])

treatments_clean = treatments_clean.drop(['given_name', 'surname'], axis=1)
all_colums = pd.Series(list(patients_clean) + list(treatments_clean))

all_colums
all_colums[all_colums.duplicated()]
# treatments_cut = pd.read_csv('treatments.csv')
# treatments_clean = pd.concat([treatments_clean, treatments_cut], ignore_index= True)

# treatments_clean
treatments_clean.hba1c_change = (treatments_clean.hba1c_start - treatments_clean.hba1c_end)

treatments_clean.head()
patients_clean.zip_code = patients_clean.zip_code.astype(str).str[:-2].str.pad(5, fillchar='0') 

patients_clean.zip_code.value_counts()
patients_clean.zip_code = patients_clean.zip_code.replace('0000n', np.nan)

patients_clean.zip_code.value_counts()
patients_clean.assigned_sex = patients_clean.assigned_sex.astype('category')

patients_clean.state = patients_clean.state.astype('category')
patients_clean.birthdate = pd.to_datetime(patients_clean.birthdate)
treatments_clean.head()
patients_clean.head()
patients_clean.height = patients_clean.replace(27, 72)
state_abbrev = {'California': 'CA', 

                'New York': 'NY', 

                'Illinois': 'IL', 

                'Florida': 'FL', 

                'Nebraska':'NE'}



def abbreviate_state(df):

    if df['state'] in state_abbrev.keys():

        abbrev = state_abbrev[df['state']]

        return abbrev

    else:

        return df['state']



patients_clean['state'] = patients_clean.apply(abbreviate_state, axis=1)



patients_clean.state.value_counts()
patients_clean.state.unique()
patients_clean.given_name = patients_clean.given_name.replace('Dsvid', 'David')
patients_clean.phone_number = patients_clean.phone_number.str.replace(r'\D+', '').str.pad(11, fillchar='1')
patients_clean = patients_clean[patients_clean.surname != 'Doe']
patients_clean = patients_clean[~(patients_clean.address.duplicated()) 

                                & patients_clean.address.notnull()]
weight_kg = patients_clean.weight.sort_values()[0]

patients_clean.loc[patients_clean.surname == 'Zaitseva', 'weight'] = weight_kg * 2.20462
patients_clean.head()
treatments_clean.head()
treatments_clean.to_csv('treatments_final.csv', index=False)
treatments_final = pd.read_csv('treatments_final.csv')

treatments_final
adverse_reactions_clean.head()
id_names = patients_clean[['patient_id', 'given_name', 'surname']]

id_names['given_name'] = id_names['given_name'].str.lower()

id_names['surname'] = id_names['surname'].str.lower()

adverse_reactions_clean = pd.merge(adverse_reactions_clean, id_names, on=['given_name', 'surname'])

# adverse_reactions_clean = adverse_reactions_clean.drop(['given_name', 'surname'], axis=1)
treatment_group = treatments_final.groupby('treatment')

treatment_group
treatment_group.mean()
Auralin = treatment_group.mean().loc['auralin', 'does_change']

Novodra = treatment_group.mean().loc['novodra', 'does_change']
treatment_group.mean().does_change
ax = treatment_group.mean().does_change.plot(kind='bar') #ax是pandas plot的返回实例，它的类是matplotlib的axes对象，里面有很多设置图的方法

ax.set_title('实验前/后胰岛素平均计量变化', fontproperties=font)#设置图表的标题，字体为中文字体

plt.show()
ax = treatment_group.mean().hba1c_change.plot(kind='bar') #ax是pandas plot的返回实例，它的类是matplotlib的axes对象，里面有很多设置图的方法

ax.set_title('实验前/后hba1c变化量均值', fontproperties=font)#设置图表的标题，字体为中文字体

plt.show()
treatment_group.mean().info()
after_adverse_reactions = treatment_group['adverse_reaction'].value_counts()

after_adverse_reactions
after_adverse_reactions.unstack()
after_adverse_reactions.unstack().plot(kind='bar', figsize=(20, 4))
ax = after_adverse_reactions.auralin.plot(kind='bar', figsize=(20, 4)) #ax是pandas plot的返回实例，它的类是matplotlib的axes对象，里面有很多设置图的方法

ax.set_title('Auralin不良反应', fontproperties=font)#设置图表的标题，字体为中文字体

plt.show()
ax = after_adverse_reactions.novodra.plot(kind='bar', figsize=(20, 4)) #ax是pandas plot的返回实例，它的类是matplotlib的axes对象，里面有很多设置图的方法

ax.set_title('Novodra不良反应', fontproperties=font)#设置图表的标题，字体为中文字体

plt.show()
ts = np.sort(treatments_final['treatment'].dropna().unique())

fig, axes = plt.subplots(1, len(ts), figsize = (24, 4), sharey=True)

for ax, t in zip(axes, ts):

    treatments_final[treatments_final['treatment'] == t]['adverse_reaction'].value_counts().sort_index().plot(kind='bar', ax=ax, title=t)
