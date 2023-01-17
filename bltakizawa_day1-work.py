%matplotlib inline 
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

np.get_printoptions()
# 取得したデータを確認する
# カラムとヘッダーとフッターのデータを確認して、概要を確認する
df_data = pd.read_csv("../input/survey.csv")
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
# 以下のコマンドでデータを実際に確認した。NaNも確認できるようにパラメータを調整
# それを確認したうえで、onehot encodingをどうするか判断した
# df_data['Gender'].value_counts(dropna=False)

# Gender
# 名寄せをして、onehot encodingする
m_df_data = df_data['Gender'].map({'Male':'Male','male':'Male','M':'Male','m':'Male','Female':'Female','female':'Female','F':'Female','f':'Female'})
df_data['Gender_str'] = m_df_data.astype(str).map(lambda x:'Gender_'+x)
df_en = pd.concat([df_data,pd.get_dummies(df_data['Gender_str'])],axis=1)

# Country
# onehot encodingする
# df_data['Country_str'] = df_data['Country'].astype(str).map(lambda x:'Country_'+x)
# df_en = pd.concat([df_en,pd.get_dummies(df_data['Country_str'])],axis=1)

print(df_en.columns)
df_en
# 各データの内容を確認し、カテゴリカル変数をonehot encodingする
# Ageは一旦そのまま
# Gender
# 名寄せをして、onehot encodingする
m_df_data = df_data['Gender'].map({'Male':'Male','male':'Male','M':'Male','m':'Male','Female':'Female','female':'Female','F':'Female','f':'Female'})
df_data['Gender_str'] = m_df_data.astype(str).map(lambda x:'Gender_'+x)
df_en = pd.concat([df_data,pd.get_dummies(df_data['Gender_str'])],axis=1)


# Country
# onehot encodingする
df_data['Country_str'] = df_data['Country'].astype(str).map(lambda x:'Country_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['Country_str'])],axis=1)

# State
# onehot encodingする
df_data['state_str'] = df_data['state'].astype(str).map(lambda x:'state_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['state_str'])],axis=1)

# self_employed
# onehot encodingする
df_data['self_employed_str'] = df_data['self_employed'].astype(str).map(lambda x:'self_employed_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['self_employed_str'])],axis=1)

# family_history
# onehot encodingする
df_data['family_history_str'] = df_data['family_history'].astype(str).map(lambda x:'family_history_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['family_history_str'])],axis=1)

# treatment
# onehot encodingする
df_data['treatment_str'] = df_data['treatment'].astype(str).map(lambda x:'treatment_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['treatment_str'])],axis=1)

# work_interfere
# onehot encodingする
df_data['work_interfere_str'] = df_data['work_interfere'].astype(str).map(lambda x:'work_interfere_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['work_interfere_str'])],axis=1)

# no_employees
# onehot encodingする
df_data['no_employees_str'] = df_data['no_employees'].astype(str).map(lambda x:'no_employees_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['no_employees_str'])],axis=1)

# remote_work
# onehot encodingする
df_data['remote_work_str'] = df_data['remote_work'].astype(str).map(lambda x:'remote_work_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['remote_work_str'])],axis=1)

# tech_company
# onehot encodingする
df_data['tech_company_str'] = df_data['tech_company'].astype(str).map(lambda x:'tech_company_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['tech_company_str'])],axis=1)

# benefits
# onehot encodingする
df_data['benefits_str'] = df_data['benefits'].astype(str).map(lambda x:'benefits_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['benefits_str'])],axis=1)

# care_options
# onehot encodingする
df_data['care_options_str'] = df_data['care_options'].astype(str).map(lambda x:'care_options_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['care_options_str'])],axis=1)

# wellness_program
# onehot encodingする
df_data['wellness_program_str'] = df_data['wellness_program'].astype(str).map(lambda x:'wellness_program_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['wellness_program_str'])],axis=1)

# seek_help
# onehot encodingする
df_data['seek_help_str'] = df_data['seek_help'].astype(str).map(lambda x:'seek_help_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['seek_help_str'])],axis=1)

# anonymity
# onehot encodingする
df_data['anonymity_str'] = df_data['anonymity'].astype(str).map(lambda x:'anonymity_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['anonymity_str'])],axis=1)

# leave
# onehot encodingする
df_data['leave_str'] = df_data['leave'].astype(str).map(lambda x:'leave_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['leave_str'])],axis=1)

# mental_health_consequence
# onehot encodingする
df_data['mental_health_consequence_str'] = df_data['mental_health_consequence'].astype(str).map(lambda x:'mental_health_consequence_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['mental_health_consequence_str'])],axis=1)

# phys_health_consequence
# onehot encodingする
df_data['phys_health_consequence_str'] = df_data['phys_health_consequence'].astype(str).map(lambda x:'phys_health_consequence_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['phys_health_consequence_str'])],axis=1)

# coworkers
# onehot encodingする
df_data['coworkers_str'] = df_data['coworkers'].astype(str).map(lambda x:'coworkers_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['coworkers_str'])],axis=1)

# supervisor
# onehot encodingする
df_data['supervisor_str'] = df_data['supervisor'].astype(str).map(lambda x:'supervisor_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['supervisor_str'])],axis=1)

# mental_health_interview
# onehot encodingする
df_data['mental_health_interview_str'] = df_data['mental_health_interview'].astype(str).map(lambda x:'mental_health_interview_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['mental_health_interview_str'])],axis=1)

# phys_health_interview
# onehot encodingする
df_data['phys_health_interview_str'] = df_data['phys_health_interview'].astype(str).map(lambda x:'phys_health_interview_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['phys_health_interview_str'])],axis=1)

# mental_vs_physical
# onehot encodingする
df_data['mental_vs_physical_str'] = df_data['mental_vs_physical'].astype(str).map(lambda x:'mental_vs_physical_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['mental_vs_physical_str'])],axis=1)

# mental_vs_physical
# onehot encodingする
df_data['obs_consequence_str'] = df_data['obs_consequence'].astype(str).map(lambda x:'obs_consequence_'+x)
df_en = pd.concat([df_en,pd.get_dummies(df_data['obs_consequence_str'])],axis=1)

df_data = df_data.drop(['Country_str','state_str','self_employed_str','family_history_str','treatment_str','work_interfere_str','no_employees_str','remote_work_str','tech_company_str','benefits_str','care_options_str','wellness_program_str','seek_help_str','anonymity_str','leave_str','mental_health_consequence_str','phys_health_consequence_str','coworkers_str','supervisor_str','mental_health_interview_str','phys_health_interview_str','mental_vs_physical_str','obs_consequence_str'],axis=1)

#特徴量エンジニアリングしたので元の変数は捨てる
df_en_fin = df_en.drop(['Gender', 'Country', 'state', 'self_employed',
       'family_history', 'treatment', 'work_interfere', 'no_employees',
       'remote_work', 'tech_company', 'benefits', 'care_options',
       'wellness_program', 'seek_help', 'anonymity', 'leave',
       'mental_health_consequence', 'phys_health_consequence', 'coworkers',
       'supervisor', 'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence','Gender_str'],axis=1)

list(df_en_fin.columns.values)
#相関行列を確認する
df_en_fin.corr().style.background_gradient().format('{:.2f}')