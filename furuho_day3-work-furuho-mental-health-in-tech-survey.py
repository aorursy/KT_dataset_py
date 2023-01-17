%matplotlib inline
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
df = pd.read_csv('../input/survey.csv')
#print(df.head())
#欠損値の確認
#df.isnull().sum()
#試行錯誤のためにコピーしておく
df_test = df.copy()
#treatment →　今回の目的
#print(df_test['treatment'].value_counts())
#Timestamp
#仮説：調査を提出した季節によって精神疾患の有無が異なる→春に自殺者が増えるらしい
#12～2:2(冬)、3～5:3(春)、6～8:0(夏)、9～11:1(秋)
#南半球では引っ繰り返るが、ここではそのまま（後で反転する）
df_test['Season'] = pd.to_datetime(df_test.Timestamp).map(lambda x:int(((x.month+6)%12)/3))
#仮説：調査を提出したのが、夜中か日中(活動時間)かで精神疾患の有無が異なる→夜中の方がリスク高いとする
#6~21時:1(日中)、22~5時:2(夜)
df_test['Time'] = pd.to_datetime(df_test.Timestamp).map(lambda y:int((((y.hour+18)%24)/16)+1))
print(df_test.head())
#Age
#print(df_test['Age'].value_counts())
#負は＋にする
df_test['Age'] = abs(df_test['Age'])
#精神疾患になる現実的な年齢(2～120歳)に限定する
df_test = df_test[(2<=df_test["Age"])&(df_test["Age"]<=120)]
#行に欠番があるのでindex番号を再振分
df_test = df_test.reset_index(drop=True)
#確認
#print(df_test['Age'].value_counts())
#gender
#全て大文字にする
gender = df_test['Gender'].str.upper()
#M:Male, F:Female, T:Trans, O:Otherに分類することとする
#上記に分類のため、まずは全種類のリストを作成してcsv保存
gender_list = gender.value_counts()
#gender_list.to_csv("gender_cntlist.csv")
mapping = { "M":"M","F":"F","MALE":"M", "FEMALE":"F", "WOMAN":"F", "MAKE":"M", "CIS MALE":"M", "MALE ":"M", "FEMALE (TRANS)":"T",
           "MAN":"M","FEMALE ":"F","QUEER":"T","ENBY":"O", "MSLE":"M","SOMETHING KINDA MALE?":"O","GUY (-ISH) ^_^":"M",
           "CIS-FEMALE/FEMME":"T","AGENDER":"O","MAIL":"M","MAILE":"M","FEMAIL":"F","NEUTER":"O",
           "OSTENSIBLY MALE, UNSURE WHAT THAT REALLY MEANS":"O","QUEER/SHE/THEY":"T","NAH":"O","FEMALE (CIS)":"F",
           "ANDROGYNE":"O","TRANS WOMAN":"T","MALE-ISH":"M","FLUID":"O","TRANS-FEMALE":"T","GENDERQUEER":"T","CIS MAN":"M",
           "CIS FEMALE":"F","A LITTLE ABOUT YOU":"O","MAL":"M","FEMAKE":"F","MALE LEANING ANDROGYNOUS":"T","MALE (CIS)":"M",
           "MALR":"M","NON-BINARY":"O", "ALL":"O", "P":"O" }
df_test['Gender'] =  gender.map(mapping)
#確認
print(df_test['Gender'].value_counts())
#数値化 仮説：F→O→T→Mの順でリスクが高い
df_test['Gender'] = df_test['Gender'].replace("F",3,regex=True)
df_test['Gender'] = df_test['Gender'].replace("O",2,regex=True)
df_test['Gender'] = df_test['Gender'].replace("T",1,regex=True)
df_test['Gender'] = df_test['Gender'].replace("M",0,regex=True)
print(df_test['Gender'].value_counts())
#Country & state
#United Statesの場合はCountryにstateを入力
country = df_test['Country'].copy()
for i in range(len(country.index)):
    if df_test.at[i, 'Country'] == 'United States':
        country[i] = "USA {0}".format(df_test.state[i])
    else:
        if not df_test.state[i] is np.nan:
            print("{0}番目の{1}のstateに{2}という不正値です".format(i, country[i], df_test.at[i,'state']))
#United States以外でstateに値が入っているものは↑で表示（特に処理はしない）
#country.value_counts().to_csv("country_cntlist.csv")
#仮説:平均気温によって、1年の気温差によって、日照時間によって、精神疾患の有無が異なる
#国＆州のデータを年間平均気温、最高/最低気温の差分、年間日照時間に置換える + 季節情報のため、南半球:1、北半球:0情報
#出典：国ごとの年間平均気温、最高/最低気温の差　http://www.data.jma.go.jp/gmd/cpd/monitor/normal/index.html
#出典：国ごとの年間日照時間 https://en.wikipedia.org/wiki/List_of_cities_by_sunshine_duration
#↑の情報を持ったcsvファイル(original)を読込み
#country_info = pd.read_csv('country_Info.csv', index_col=0)
#↓直接入力
city = pd.Series(['United Kingdom','USA CA','Canada','USA WA','USA NY','USA TN',
               'Germany','USA TX','USA OR','USA PA','USA OH','USA IL','Netherlands',
               'USA IN','Ireland','USA MI','USA MN','Australia','USA MA','USA FL',
               'USA NC','USA VA','France','USA MO','USA GA','USA WI','USA nan',
               'India','USA UT','USA CO','New Zealand','Poland','Italy','USA MD',
               'Switzerland','USA AL','USA AZ','Sweden','Brazil','Belgium',
               'South Africa','USA OK','USA NJ','Israel','USA SC','USA KY',
               'Singapore','Bulgaria','USA IA','USA DC','USA CT','Finland','USA VT',
               'Mexico','USA SD','Russia','Austria','USA KS','USA NH','USA NV',
               'USA NE','USA NM','Portugal','Denmark','Croatia','USA WY','Colombia',
               'Greece','Moldova','China','Norway','Nigeria','USA ME','Slovenia',
               'USA MS','Spain','Uruguay','Georgia','Japan','Philippines',
               'Bosnia and Herzegovina','Hungary','Thailand','Romania','USA WV',
               'Bahamas, The','Czech Republic','Latvia','USA LA','USA RI',
               'Costa Rica','USA ID'])
tempa = pd.Series([11.8,17.3,6.5,11.5,13.2,15.5,10,20.9,12.5,13.4,11.6,9.9,10.1,
                11.8,9.8,10.2,8,18.2,10.9,20.3,16.1,14.8,11.7,12.7,17,9.2,14.5,
                25.2,11.6,10.4,13.1,8.4,15.5,13.2,9.4,18.5,23.9,6.7,23.9,10.6,
                16.8,15.9,12.3,15.9,17.8,14.4,27.6,10.3,10.4,14.5,10.3,5.3,7.7,
                16.7,8,5.8,10.5,13.9,7.9,20.3,10.8,14.2,17.2,9.1,12.1,8.1,13.4,
                18.8,10.2,12.9,4.8,26.7,8,9.7,18.3,15,16.3,13.3,15.4,27.7,10.4,
                11,28.9,10.8,13.1,25.2,8.4,6.2,20.7,10.9,22.7,11.1])
tempd = pd.Series([13,7,31.1,14,24.3,23,18.9,17.3,16.2,24.7,25.4,27.9,14.8,25.9,
                10.2,26.3,32,10.4,24.6,16.1,21,22.9,15.7,27,20.5,26.9,24.3,19.1,
                27.4,24.1,8.7,21,18.1,24.1,18.2,19.2,21.8,20.8,5.5,15.1,8.9,
                24.3,23,15.6,20.5,24.1,1.8,21.7,29.8,24.3,25.9,23.4,28.3,5.3,
                31,25.9,20.5,27.2,26.9,24.9,27.2,23.3,11.7,16.7,20.9,23,0.8,
                18.5,24.1,29.8,21.5,3.4,25.5,21.5,19.7,19.3,11.8,23,21.2,3.4,
                20.3,22.4,4.3,24.5,22.3,6.9,19.6,23.4,16.6,24,1.9,25])
sunH = pd.Series([1633,3254,2051,2169,2534,2510,1626,2577,2340,2498,2182,2508,
               1662,2440,1453,2435,2710,2635,2633,2879,2821,2829,1662,2809,
               2738,2483,2527,2684,3029,3106,2058,1571,2873,2581,1566,2738,
               3871,1821,2181,1546,3094,3089,2498,3311,2821,2514,2022,2177,
               2726,2527,2633,1858,2051,2555,2710,1901,1884,2922,2633,3825,
               2726,3415,2806,1539,1913,3106,1328,2848,2135,2670,1668,1845,
               2633,2020,2888,2769,2481,2207,1876,2103,1876,1988,2629,2115,
               2182,2575,1668,1635,2648,2633,3055,2993])
pos = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,
              0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
country_i = pd.DataFrame({'Country':city, 'TempAve':tempa, 'TempDif':tempd, 'SunHours':sunH, 'Pos':pos})
country_info = country_i.set_index('Country')
#print(country_info)
#上記データを入力
for i in range(len(country.index)):
    df_test.at[i,'TempAve'] = country_info.at[country[i], "TempAve"]
    df_test.at[i,'TempDif'] = country_info.at[country[i], "TempDif"]
    df_test.at[i,'SunHours'] = country_info.at[country[i], "SunHours"]
    # 2(冬)⇔ 0(夏)、 3(春)⇔ 1(秋)
    if country_info.at[country[i], "Pos"] == 1:
        df_test.at[i,'Season'] = (df_test.at[i,'Season']+2)%4
#print(df_test.head())
#self employed
#仮定：欠損(未記入)は、No(自営業じゃない)
df_test['self_employed'] = df_test['self_employed'].fillna('No')
#print(df_test['self_employed'].value_counts())
#work interfere
#仮定：欠損(未記入)は、Never(妨げになったことがない)
#仮定：重症度　Often:3 ＞ Sometimes:2 ＞ Rarely:1 ＞ Never:0 とする
df_test['work_interfere'] = df_test['work_interfere'].fillna('Never')
df_test['work_interfere'] = df_test['work_interfere'].replace('Often',3,regex=True)
df_test['work_interfere'] = df_test['work_interfere'].replace('Sometimes',2,regex=True)
df_test['work_interfere'] = df_test['work_interfere'].replace('Rarely',1,regex=True)
df_test['work_interfere'] = df_test['work_interfere'].replace('Never',0,regex=True)
#print(df_test['work_interfere'].value_counts())
#No employees
df_test['no_employees'] = df_test['no_employees'].replace('More than 1000',5,regex=True)
df_test['no_employees'] = df_test['no_employees'].replace('500-1000',4,regex=True)
df_test['no_employees'] = df_test['no_employees'].replace('100-500',3,regex=True)
df_test['no_employees'] = df_test['no_employees'].replace('26-100',2,regex=True)
df_test['no_employees'] = df_test['no_employees'].replace('6-25',1,regex=True)
df_test['no_employees'] = df_test['no_employees'].replace('1-5',0,regex=True)
#print(df_test['no_employees'].value_counts())
#Leave
df_test['leave'] = df_test['leave'].replace('Very difficult',4,regex=True)
df_test['leave'] = df_test['leave'].replace('Somewhat difficult',3,regex=True)
df_test['leave'] = df_test['leave'].replace('Somewhat easy',2,regex=True)
df_test['leave'] = df_test['leave'].replace('Very easy',1,regex=True)
df_test['leave'] = df_test['leave'].replace("Don't know",0,regex=True)
#print(df_test['leave'].value_counts())
#comments
#仮説：コメントが有り(c1)/無し(c0)で精神疾患の有無が異なる
df_test['comments_bin'] = df['comments'].map(lambda z:(z is not np.nan )*1)
#print(df_test.head())
#別の列にしたものは削除
df_test = df_test.drop(['Timestamp','Country','state','comments'],axis=1)
#グラフにして確認
#for col_name in df_test.columns:
#    if (col_name != 'treatment') & (col_name != 'Age') & (col_name != 'SunHours') & (col_name != 'TempAve') & (col_name != 'TempDif'):
#        Allplot = pd.crosstab(index=df_test['treatment'], columns=df_test[col_name], margins=False)
#        Allplot.plot.bar()
#        plt.show()
#傾向より変更＆追加
df_test["m_vs_h_know"] = df_test.apply(lambda x: 0 if x.mental_vs_physical == "Don't know" else 1, axis=1)
#数値データはYes,Noで分けてプロット
fig = plt.figure(figsize=(10,20))
#Age
AgeY = df_test.apply(lambda x: x.Age if x.treatment == 'Yes' else np.nan, axis=1)
AgeN = df_test.apply(lambda x: x.Age if x.treatment == 'No' else np.nan, axis=1)
upper = (int(df_test['Age'].max()/10+1))*10
#subplot = fig.add_subplot(4,1,1)
#subplot.set_title('Age')
#subplot.hist([AgeY, AgeN],bins=int(upper/5),range=(0, upper), label=['Yes', 'No'])
#subplot.legend()
#TempAve
TempAY = df_test.apply(lambda x: x.TempAve if x.treatment == 'Yes' else np.nan, axis=1)
TempAN = df_test.apply(lambda x: x.TempAve if x.treatment == 'No' else np.nan, axis=1)
upper = (int(df_test['TempAve'].max()/5+1))*5
#subplot = fig.add_subplot(4,1,2)
#subplot.set_title('Temperature Average')
#subplot.hist([TempAY, TempAN],bins=int(upper/2.5),range=(0, upper), label=['Yes', 'No'])
#subplot.legend()
#TempDif
TempDY = df_test.apply(lambda x: x.TempDif if x.treatment == 'Yes' else np.nan, axis=1)
TempDN = df_test.apply(lambda x: x.TempDif if x.treatment == 'No' else np.nan, axis=1)
upper = (int(df_test['TempDif'].max()/5+1))*5
#subplot = fig.add_subplot(4,1,3)
#subplot.set_title('Temperature Difference')
#subplot.hist([TempDY, TempDN],bins=int(upper/2.5),range=(0, upper), label=['Yes', 'No'])
#subplot.legend()
#SunshineHours
SunHY = df_test.apply(lambda x: x.SunHours if x.treatment == 'Yes' else np.nan, axis=1)
SunHN = df_test.apply(lambda x: x.SunHours if x.treatment == 'No' else np.nan, axis=1)
upper = (int(df_test['SunHours'].max()/50+1))*50
#subplot = fig.add_subplot(4,1,4)
#subplot.set_title('Sunshine Hours')
#subplot.hist([SunHY, SunHN],bins=int((upper-1000)/250),range=(1000, upper), label=['Yes', 'No'])
#subplot.legend()
#傾向より追加
#df_test["Age>30"] = df_test.apply(lambda x: 0 if x.Age < 30 else 1, axis=1)
#df_test["SunHours>2000"] = df_test.apply(lambda x: 0 if x.SunHours <2000.0 else 1, axis=1)
#カテゴリー変数はダミー変数に
df_dummy = pd.get_dummies(df_test)
#yes/noの2択はyesだけに絞る
df_dummy = df_dummy.drop(['self_employed_No','family_history_No','treatment_No',
                          'remote_work_No','tech_company_No','obs_consequence_No'],axis=1)
#print(df_dummy.head())
#傾向より追加→グラフで確認
df_dummy["m_h_DontKnow"] = df_dummy["benefits_Don't know"] + df_dummy["care_options_Not sure"] + df_dummy["wellness_program_Don't know"] + df_dummy["seek_help_Don't know"] + df_dummy["anonymity_Don't know"]
df_dummy["m_h_Yes"] = df_dummy["benefits_Yes"] + df_dummy["care_options_Yes"] + df_dummy["wellness_program_Yes"] + df_dummy["seek_help_Yes"] + df_dummy["anonymity_Yes"]
fig = plt.figure(figsize=(10,10))
#Don't know
upper = 5
dkY = df_dummy.apply(lambda x: x.m_h_DontKnow if x.treatment_Yes == 1 else np.nan, axis=1)
dkN = df_dummy.apply(lambda x: x.m_h_DontKnow if x.treatment_Yes == 0 else np.nan, axis=1)
#subplot = fig.add_subplot(2,1,1)
#subplot.set_title('MH_DontKnow')
#subplot.hist([dkY, dkN],bins=upper,range=(0, upper), label=['Yes', 'No'])
#subplot.legend()
#Yes
YY = df_dummy.apply(lambda x: x.m_h_Yes if x.treatment_Yes == 1 else np.nan, axis=1)
YN = df_dummy.apply(lambda x: x.m_h_Yes if x.treatment_Yes == 0 else np.nan, axis=1)
#subplot = fig.add_subplot(2,1,2)
#subplot.set_title('MH_Yes')
#subplot.hist([YY, YN],bins=upper,range=(0, upper), label=['Yes', 'No'])
#subplot.legend()
#傾向より追加
#df_dummy["m_h_DontKnow_bin"] = df_dummy["m_h_DontKnow"].map(lambda x: 0 if x<2 else 1)
#df_dummy["m_h_Yes_bin"] = df_dummy["m_h_Yes"].map(lambda x: 0 if x<1 else 1)
cols = ['treatment_Yes']  + [col for col in df_dummy if col != 'treatment_Yes']
df_dummy = df_dummy[cols]
#df_dummy.corr().style.background_gradient().format('{:.2f}')
#色々被っているデータを削除
df_dummy_lim = df_dummy.drop(["benefits_Don't know","benefits_Yes","benefits_No",
 "care_options_Not sure","care_options_Yes","care_options_No",
 "wellness_program_Don't know","wellness_program_Yes","wellness_program_No",
 "seek_help_Don't know","seek_help_Yes","seek_help_No",
 "anonymity_Don't know","anonymity_Yes","anonymity_No",
 "mental_health_consequence_Maybe","mental_health_consequence_Yes","mental_health_consequence_No",
 "phys_health_consequence_Maybe","phys_health_consequence_Yes",
 "coworkers_Some of them","coworkers_Yes","supervisor_Some of them","supervisor_No",
 "mental_health_interview_Maybe","mental_health_interview_Yes",
 "phys_health_interview_Maybe","phys_health_interview_Yes",
 "mental_vs_physical_Don't know","mental_vs_physical_Yes", "m_h_Yes"],axis=1) #"m_h_DontKnow"
df_dummy_lim.corr().style.background_gradient().format('{:.2f}')
#VIF関数用
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
# VIFを取得
max_v = len(df_dummy_lim.columns)
vif = pd.DataFrame(np.zeros([max_v, 3]), columns=['R^2cof','VIF1','VIF2'])
df_vif = df_dummy_lim.drop(["treatment_Yes"],axis=1)
for i in range(max_v-1):
    y=df_vif[df_vif.columns[i]]
    X=df_vif.drop(df_vif.columns[i], axis=1)
    regr = LinearRegression(fit_intercept=True)
    regr.fit(X, y)
    vif.at[i,"R^2cof"] = np.power(regr.score(X,y),2)
    if(vif.at[i,"R^2cof"] != 1):
        #１：単回帰分析を回してVIFを計算する
        vif.at[i,"VIF1"] = 1/(1 - vif.at[i,"R^2cof"])
        #２：関数を使って、VIFを求める方法
        vif.at[i,"VIF2"] = variance_inflation_factor(df_dummy_lim.values, i+1)
    else:
        print(df_vif.columns[i],X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
        vif.at[i,"VIF1"] = np.nan
        vif.at[i,"VIF2"] = np.nan
print(vif)
#それぞれ分けた性別と季節の組合せで引っかかっているので、両方とも代表の一つ以外は削除
#df_dummy_last = df_dummy_lim.drop(["Gender_F","Gender_O","Gender_T","Season_S1","Season_S3","Season_S4"],axis=1)
df_dummy_last = df_dummy_lim.copy()
#一応、確認
#max_v = len(df_dummy_last.columns)
#Re_vif = pd.DataFrame(np.zeros([max_v, 3]), columns=['R^2cof','VIF1','VIF2'])
#Re_df_vif = df_dummy_last.drop(["treatment_Yes"],axis=1)
#for i in range(max_v-1):
#    y=Re_df_vif[Re_df_vif.columns[i]]
#    X=Re_df_vif.drop(Re_df_vif.columns[i], axis=1)
#    regr = LinearRegression(fit_intercept=True)
#    regr.fit(X, y)
#    Re_vif.at[i,"R^2cof"] = np.power(regr.score(X,y),2)
#    if(Re_vif.at[i,"R^2cof"] != 1):
#        #１：単回帰分析を回してVIFを計算する
#        Re_vif.at[i,"VIF1"] = 1/(1 - Re_vif.at[i,"R^2cof"])
#        #２：関数を使って、VIFを求める方法
#        Re_vif.at[i,"VIF2"] = variance_inflation_factor(df_dummy_last.values, i+1)
#    else:
#        print(Re_df_vif.columns[i],X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
#        Re_vif.at[i,"VIF1"] = np.nan
#        Re_vif.at[i,"VIF2"] = np.nan
#print(Re_vif)
#分割用＆ハイパーパラメータ探索用
from sklearn.model_selection import train_test_split, GridSearchCV
#評価用
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
# データの準備  とりあえず、トレーニング7割：テスト3割
X = df_dummy_last.drop(['treatment_Yes'], axis=1)
y = df_dummy_last.treatment_Yes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.ensemble import RandomForestClassifier
#次にハイパーパラメータ探索
param_grid = {'criterion':["gini"],
              'n_estimators':[30,33,35],
              'max_depth':[8,10,12],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[2,3]}
CVf=10
Forestcv = GridSearchCV(RandomForestClassifier(random_state=100),
                        param_grid=param_grid,cv=CVf,n_jobs=-1)
Forestcv = Forestcv.fit(X_train, y_train)
y_forestcv = Forestcv.predict(X_test)
print("score_train= {:.5f}".format(Forestcv.score(X_train, y_train)))
print("score_test = {:.5f}".format(Forestcv.score(X_test, y_test)))
print(Forestcv.best_params_)
from sklearn.linear_model import LinearRegression,LogisticRegression
#liblinear
param_grid = {'tol':[1e-1, 1e-2],
              'C':[3.0, 4.0, 5.0],
              'intercept_scaling':[3.0,3.2,3.4,3.6,3.8]}
#cv=5~420まで試したが、10くらいで頭打ち
CVl=10
LrLcv = GridSearchCV(LogisticRegression(random_state=100,
                                        fit_intercept=True,
                                        solver='liblinear'),
                      param_grid=param_grid,cv=CVl,n_jobs=-1)
LrLcv = LrLcv.fit(X_train, y_train)
y_lrLcv = LrLcv.predict(X_test)
print("score_train= {:.5f}".format(LrLcv.score(X_train, y_train)))
print("score_test = {:.5f}".format(LrLcv.score(X_test, y_test)))
#liblinear以外
param_grid = {'tol':[1e-1, 1e-2],
              'C':[0.6,0.8,1.0,1.2],
              'max_iter':[20, 50, 100], #小さいとConvergenceWarningが出る
              'solver':['newton-cg','lbfgs']}#'sag'と'saga'は大きいデータ用
#cvは共通
CVl=10
LrNLcv = GridSearchCV(LogisticRegression(random_state=100),
                      param_grid=param_grid,cv=CVl,n_jobs=-1)
LrNLcv = LrNLcv.fit(X_train, y_train)
y_lrNLcv = LrNLcv.predict(X_test)
print("score_train= {:.5f}".format(LrNLcv.score(X_train, y_train)))
print("score_test = {:.5f}".format(LrNLcv.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier, export_graphviz
param_grid = {'criterion':["gini", "entropy"],
              #'splitter':["best", "random"], #変わらない
              'max_depth':[3,5,7],
              #'min_samples_split':[4,6,8,10], #変わらない
              'min_samples_leaf':[3,12,14]}
#色々試した結果CV=50
CVt=50
Treecv = GridSearchCV(DecisionTreeClassifier(random_state=100),
                      param_grid=param_grid,cv=CVt,n_jobs=-1)
Treecv = Treecv.fit(X_train, y_train)
y_treecv = Treecv.predict(X_test)
print("score_train= {:.5f}".format(Treecv.score(X_train, y_train)))
print("score_test = {:.5f}".format(Treecv.score(X_test, y_test)))
from sklearn.ensemble import AdaBoostClassifier
param_grid = {'base_estimator__criterion':["gini", "entropy"],
              'base_estimator__splitter':["best", "random"],
              'base_estimator__max_depth':[5,7,9],
              'base_estimator__min_samples_leaf':[10,12,14],
              'base_estimator__min_samples_split':[2,3,4],
              'n_estimators':[1,2]}
CVa=20
Boostcv = GridSearchCV(AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state=100)),
                                          param_grid=param_grid,cv=CVa,n_jobs=-1)
Boostcv = Boostcv.fit(X_train, y_train)
y_boostcv = Boostcv.predict(X_test)
print("score_train= {:.5f}".format(Boostcv.score(X_train, y_train)))
print("score_test = {:.5f}".format(Boostcv.score(X_test, y_test)))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
#標準化
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

CVk=20
param_grid = {'n_neighbors':[5,10,12,13,14,15],
              #'algorithm':['auto','ball_tree','kd_tree','brute'],#変わらない
              #'leaf_size':[10,20,30],#変わらない
              'p':[1,2]}
KNeighborcv = GridSearchCV(KNeighborsClassifier(),
                           param_grid=param_grid,cv=CVk,n_jobs=-1)

KNeighborcv.fit(X_train_std, y_train)
y_kneighborcv = KNeighborcv.predict(X_test_std)
print("score_train= {:.5f}".format(KNeighborcv.score(X_train_std, y_train)))
print("score_test = {:.5f}".format(KNeighborcv.score(X_test_std, y_test)))
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler #標準化用
# 標準化
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)
#適当パラメータ
C = 5
kernel = "rbf"  #rbf:ガウス、poly:多項式、sigmoid:シグモイド、もう一つある。。。
gamma = 1
#gammaが小さいと線形、大きいとデータに特異的(インパルス)
Svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=100)
Svm = Svm.fit(X_train, y_train)
y_svm = Svm.predict(X_test)
print("score_train= {:.5f}".format(Svm.score(X_train, y_train)))
print("score_test = {:.5f}".format(Svm.score(X_test, y_test)))
#↑汎化性が低い。。
#ハイパラメータ
CVs = 5
parameters = {'kernel':['sigmoid', 'rbf'], 'C':[0.02, 0.05, 0.1, 0.2], 'gamma':[0.02, 0.05, 0.1, 0.2]}
SvmcvSig = GridSearchCV(SVC(random_state=100), parameters, cv=CVs)
SvmcvSig.fit(X_train, y_train)
y_svmcvsig = SvmcvSig.predict(X_test)
print("score_train= {:.5f}".format(SvmcvSig.score(X_train, y_train)))
print("score_test = {:.5f}".format(SvmcvSig.score(X_test, y_test)))
print(SvmcvSig.best_params_)
parameters = {'C':[0.05, 0.1, 0.2], 'gamma':[0.05, 0.1, 0.2],  'degree':[2,3,5,10],
              'coef0':[0.0, 1.0, -1.0] }
SvmcvPoly = GridSearchCV(SVC(random_state=100, kernel='poly'), parameters, cv=CVs)
SvmcvPoly.fit(X_train, y_train)
y_svmcvpoly = SvmcvPoly.predict(X_test)
print("score_train= {:.5f}".format(SvmcvPoly.score(X_train, y_train)))
print("score_test = {:.5f}".format(SvmcvPoly.score(X_test, y_test)))
print(SvmcvPoly.best_params_)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD,RMSprop, Adagrad, Adadelta, Adam
from keras.utils import np_utils
from sklearn import metrics
# one-hotベクトルに変換
y_train_o = np_utils.to_categorical(y_train)
y_test_o = np_utils.to_categorical(y_test)
#グラフ描画用
def graph(nndf):
    nndf[["loss", "val_loss"]].plot()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    nndf[["acc", "val_acc"]].plot()
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.ylim([0,1.0])
    plt.show()
n = X_train.shape[1]

#モデル選択　lr:学習率
lr=0.01
sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
# rms = RMSprop(lr=0.01)
# adag = Adagrad(lr=0.01)
# adad = Adadelta(lr=0.01)
# adam = Adam(lr=0.01)

#epochs:パラメータの更新回数(全データを使うまでが1epoch)
#batchsize:1回の更新に使うデータサイズ
#適当
ep = 40
bs = 128
#ノード数は適当
model1 = Sequential()
#Sequentialのモデルに対して追加していく
model1.add(Dense(12, activation='relu', input_dim=24))
model1.add(Dense(6, activation='relu', input_dim=12))
model1.add(Dense(2, activation='softmax')) #softmax:0~1の間に値を抑えるため
model1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#categorical_crossentropy:クラス分類するときの誤差関数
# 計算の実行
nn_sgd1 = model1.fit(X_train, y_train_o, epochs=ep, batch_size=bs, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_sgd1.history)
# グラフ化
graph(nndf)
#確認　値が毎回実行するたびに変わる、、、
y_nn_sgd1_  = np.argmax(model1.predict(X_test),axis=1)
y_nn_sgd1   = model1.predict_classes(X_test, batch_size=bs, verbose=1)
print(metrics.classification_report(y_test, y_nn_sgd1))
#confusion matrix を2通りのやり方で算出　→　結果は一緒
c1=metrics.confusion_matrix(y_test, y_nn_sgd1)
print("{}".format((c1[0,0]+c1[1,1])/sum(sum(c1))))
c=confusion_matrix(y_test, y_nn_sgd1_)
print("{}".format((c[0,0]+c[1,1])/sum(sum(c))))
#ノード数は適当
model3 = Sequential()
model3.add(Dense(36, activation='relu', input_dim=24))
model3.add(Dense(24, activation='relu', input_dim=36))
model3.add(Dense(12, activation='relu', input_dim=24))
model3.add(Dense(6, activation='relu', input_dim=12))
model3.add(Dense(2, activation='softmax'))
model3.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# 計算の実行
nn_sgd3 = model3.fit(X_train, y_train_o, epochs=ep, batch_size=bs, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_sgd3.history)
# グラフ化
graph(nndf)
#確認　値が毎回実行するたびに変わる、、、
y_nn_sgd3   = model3.predict_classes(X_test, batch_size=bs, verbose=1)
print(metrics.classification_report(y_test, y_nn_sgd3))
c3=metrics.confusion_matrix(y_test, y_nn_sgd3)
print("{}".format((c3[0,0]+c3[1,1])/sum(sum(c3))))
#ノード数は適当
model5 = Sequential()
model5.add(Dense(36, activation='relu', input_dim=24))
model5.add(Dense(48, activation='relu', input_dim=36))
model5.add(Dense(36, activation='relu', input_dim=48))
model5.add(Dense(24, activation='relu', input_dim=36))
model5.add(Dense(12, activation='relu', input_dim=24))
model5.add(Dense(6, activation='relu', input_dim=12))
model5.add(Dense(2, activation='softmax'))
model5.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# 計算の実行
nn_sgd5 = model5.fit(X_train, y_train_o, epochs=ep, batch_size=bs, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_sgd5.history)
# グラフ化
graph(nndf)
#確認　値が毎回実行するたびに変わる、、、
y_nn_sgd5   = model5.predict_classes(X_test, batch_size=bs, verbose=1)
print(metrics.classification_report(y_test, y_nn_sgd5))
c5=metrics.confusion_matrix(y_test, y_nn_sgd5)
print("{}".format((c5[0,0]+c5[1,1])/sum(sum(c5))))
#ノード数は適当
model7 = Sequential()
model7.add(Dense(36, activation='relu', input_dim=24))
model7.add(Dense(48, activation='relu', input_dim=36))
model7.add(Dense(60, activation='relu', input_dim=48))
model7.add(Dense(48, activation='relu', input_dim=60))
model7.add(Dense(36, activation='relu', input_dim=48))
model7.add(Dense(24, activation='relu', input_dim=36))
model7.add(Dense(12, activation='relu', input_dim=24))
model7.add(Dense(6, activation='relu', input_dim=12))
model7.add(Dense(2, activation='softmax'))
model7.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# 計算の実行
nn_sgd7 = model7.fit(X_train, y_train_o, epochs=ep, batch_size=bs, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_sgd7.history)
# グラフ化
graph(nndf)
#確認　値が毎回実行するたびに変わる、、、
y_nn_sgd7   = model7.predict_classes(X_test, batch_size=bs, verbose=1)
print(metrics.classification_report(y_test, y_nn_sgd7))
c7=metrics.confusion_matrix(y_test, y_nn_sgd7)
print("{}".format((c7[0,0]+c7[1,1])/sum(sum(c7))))
#ノード数は適当
model9 = Sequential()
model9.add(Dense(36, activation='relu', input_dim=24))
model9.add(Dense(48, activation='relu', input_dim=36))
model9.add(Dense(60, activation='relu', input_dim=48))
model9.add(Dense(72, activation='relu', input_dim=60))
model9.add(Dense(60, activation='relu', input_dim=72))
model9.add(Dense(48, activation='relu', input_dim=60))
model9.add(Dense(36, activation='relu', input_dim=48))
model9.add(Dense(24, activation='relu', input_dim=36))
model9.add(Dense(12, activation='relu', input_dim=24))
model9.add(Dense(6, activation='relu', input_dim=12))
model9.add(Dense(2, activation='softmax'))
model9.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# 計算の実行
nn_sgd9 = model9.fit(X_train, y_train_o, epochs=ep, batch_size=bs, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_sgd9.history)
# グラフ化
graph(nndf)
#確認　値が毎回実行するたびに変わる、、、
y_nn_sgd9   = model9.predict_classes(X_test, batch_size=bs, verbose=1)
print(metrics.classification_report(y_test, y_nn_sgd9))
c9=metrics.confusion_matrix(y_test, y_nn_sgd9)
print("{}".format((c9[0,0]+c9[1,1])/sum(sum(c9))))
#色々試したが、以下がよさそう
lrR = 0.001
rms = RMSprop(lr=lrR)
epR = 100
bsR = 64
modelr3 = Sequential()
modelr3.add(Dense(36, activation='relu', input_dim=24))
modelr3.add(Dense(24, activation='relu', input_dim=36))
modelr3.add(Dense(12, activation='relu', input_dim=24))
modelr3.add(Dense(6, activation='relu', input_dim=12))
modelr3.add(Dense(2, activation='softmax'))
modelr3.compile(loss='categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])
# 計算の実行
nn_rms3 = modelr3.fit(X_train, y_train_o, epochs=epR, batch_size=bsR, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_rms3.history)
# グラフ化
graph(nndf)
y_nn_rms3   = modelr3.predict_classes(X_test, batch_size=bsR, verbose=1)
print(metrics.classification_report(y_test, y_nn_rms3))
cR3=metrics.confusion_matrix(y_test, y_nn_rms3)
print("{}".format((cR3[0,0]+cR3[1,1])/sum(sum(cR3))))
modelr5 = Sequential()
modelr5.add(Dense(36, activation='relu', input_dim=24))
modelr5.add(Dense(48, activation='relu', input_dim=36))
modelr5.add(Dense(36, activation='relu', input_dim=48))
modelr5.add(Dense(24, activation='relu', input_dim=36))
modelr5.add(Dense(12, activation='relu', input_dim=24))
modelr5.add(Dense(6, activation='relu', input_dim=12))
modelr5.add(Dense(2, activation='softmax'))
modelr5.compile(loss='categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])
# 計算の実行
nn_rms5 = modelr5.fit(X_train, y_train_o, epochs=epR, batch_size=bsR, validation_data=(X_test, y_test_o))
# 各epochにおける損失と精度をdfに入れる
nndf = pd.DataFrame(nn_rms5.history)
# グラフ化
graph(nndf)
y_nn_rms5   = modelr5.predict_classes(X_test, batch_size=bsR, verbose=1)
print(metrics.classification_report(y_test, y_nn_rms5))
cR5=metrics.confusion_matrix(y_test, y_nn_rms5)
print("{}".format((cR5[0,0]+cR5[1,1])/sum(sum(cR5))))
lrAG = 0.01
adag = Adagrad(lr=lrAG)
epAG = 200
bsAG = 128
modelag1 = Sequential()
modelag1.add(Dense(12, activation='relu', input_dim=24))
modelag1.add(Dense(6, activation='relu', input_dim=12))
modelag1.add(Dense(2, activation='softmax'))
modelag1.compile(loss='categorical_crossentropy',
              optimizer=adag,
              metrics=['accuracy'])
# 計算の実行
nn_ag1 = modelag1.fit(X_train, y_train_o, epochs=epAG, batch_size=bsAG, validation_data=(X_test, y_test_o))
nndf = pd.DataFrame(nn_ag1.history)
# グラフ化
graph(nndf)
y_nn_ag1 = modelag1.predict_classes(X_test, batch_size=bsAG, verbose=1)
print(metrics.classification_report(y_test, y_nn_ag1))
cag1=metrics.confusion_matrix(y_test, y_nn_ag1)
print("{}".format((cag1[0,0]+cag1[1,1])/sum(sum(cag1))))
modelag3 = Sequential()
modelag3.add(Dense(36, activation='relu', input_dim=24))
modelag3.add(Dense(24, activation='relu', input_dim=36))
modelag3.add(Dense(12, activation='relu', input_dim=24))
modelag3.add(Dense(6, activation='relu', input_dim=12))
modelag3.add(Dense(2, activation='softmax'))
modelag3.compile(loss='categorical_crossentropy',
              optimizer=adag,
              metrics=['accuracy'])
# 計算の実行
nn_ag3 = modelag3.fit(X_train, y_train_o, epochs=epAG, batch_size=bsAG, validation_data=(X_test, y_test_o))
nndf = pd.DataFrame(nn_ag3.history)
# グラフ化
graph(nndf)
y_nn_ag3   = modelag3.predict_classes(X_test, batch_size=bsAG, verbose=1)
print(metrics.classification_report(y_test, y_nn_ag3))
cag3=metrics.confusion_matrix(y_test, y_nn_ag3)
print("{}".format((cag3[0,0]+cag3[1,1])/sum(sum(cag3))))
lrAD = 0.01
adad = Adadelta(lr=lrAD)
epAD = 800
bsAD = 128
modelad7 = Sequential()
modelad7.add(Dense(36, activation='relu', input_dim=24))
modelad7.add(Dense(48, activation='relu', input_dim=36))
modelad7.add(Dense(60, activation='relu', input_dim=48))
modelad7.add(Dense(48, activation='relu', input_dim=60))
modelad7.add(Dense(36, activation='relu', input_dim=48))
modelad7.add(Dense(24, activation='relu', input_dim=36))
modelad7.add(Dense(12, activation='relu', input_dim=24))
modelad7.add(Dense(6, activation='relu', input_dim=12))
modelad7.add(Dense(2, activation='softmax'))
modelad7.compile(loss='categorical_crossentropy',
              optimizer=adad,
              metrics=['accuracy'])
# 計算の実行
nn_ad7 = modelad7.fit(X_train, y_train_o, epochs=epAD, batch_size=bsAD, validation_data=(X_test, y_test_o))
nndf = pd.DataFrame(nn_ad7.history)
# グラフ化
graph(nndf)
y_nn_ad7   = modelad7.predict_classes(X_test, batch_size=bsAD, verbose=1)
print(metrics.classification_report(y_test, y_nn_ad7))
cad7=metrics.confusion_matrix(y_test, y_nn_ad7)
print("{}".format((cad7[0,0]+cad7[1,1])/sum(sum(cad7))))
modelad9 = Sequential()
modelad9.add(Dense(36, activation='relu', input_dim=24))
modelad9.add(Dense(48, activation='relu', input_dim=36))
modelad9.add(Dense(60, activation='relu', input_dim=48))
modelad9.add(Dense(72, activation='relu', input_dim=60))
modelad9.add(Dense(60, activation='relu', input_dim=72))
modelad9.add(Dense(48, activation='relu', input_dim=60))
modelad9.add(Dense(36, activation='relu', input_dim=48))
modelad9.add(Dense(24, activation='relu', input_dim=36))
modelad9.add(Dense(12, activation='relu', input_dim=24))
modelad9.add(Dense(6, activation='relu', input_dim=12))
modelad9.add(Dense(2, activation='softmax'))
modelad9.compile(loss='categorical_crossentropy',
              optimizer=adad,
              metrics=['accuracy'])
# 計算の実行
nn_ad9 = modelad9.fit(X_train, y_train_o, epochs=epAD, batch_size=bsAD, validation_data=(X_test, y_test_o))
nndf = pd.DataFrame(nn_ad9.history)
# グラフ化
graph(nndf)
y_nn_ad9   = modelad9.predict_classes(X_test, batch_size=bsAD, verbose=1)
print(metrics.classification_report(y_test, y_nn_ad9))
cad9=metrics.confusion_matrix(y_test, y_nn_ad9)
print("{}".format((cad9[0,0]+cad9[1,1])/sum(sum(cad9))))
lrAM = 0.001
adam = Adam(lr=lrAM)
epAM = 80
bsAM = 128
modelam3 = Sequential()
modelam3.add(Dense(36, activation='relu', input_dim=24))
modelam3.add(Dense(24, activation='relu', input_dim=36))
modelam3.add(Dense(12, activation='relu', input_dim=24))
modelam3.add(Dense(6, activation='relu', input_dim=12))
modelam3.add(Dense(2, activation='softmax'))
modelam3.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
# 計算の実行
nn_am3 = modelam3.fit(X_train, y_train_o, epochs=epAM, batch_size=bsAM, validation_data=(X_test, y_test_o))
nndf = pd.DataFrame(nn_am3.history)
# グラフ化
graph(nndf)
y_nn_am3   = modelam3.predict_classes(X_test, batch_size=bsAM, verbose=1)
print(metrics.classification_report(y_test, y_nn_am3))
cam3=metrics.confusion_matrix(y_test, y_nn_am3)
print("{}".format((cam3[0,0]+cam3[1,1])/sum(sum(cam3))))
modelam7 = Sequential()
modelam7.add(Dense(36, activation='relu', input_dim=24))
modelam7.add(Dense(48, activation='relu', input_dim=36))
modelam7.add(Dense(60, activation='relu', input_dim=48))
modelam7.add(Dense(48, activation='relu', input_dim=60))
modelam7.add(Dense(36, activation='relu', input_dim=48))
modelam7.add(Dense(24, activation='relu', input_dim=36))
modelam7.add(Dense(12, activation='relu', input_dim=24))
modelam7.add(Dense(6, activation='relu', input_dim=12))
modelam7.add(Dense(2, activation='softmax'))
modelam7.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
# 計算の実行
nn_am7 = modelam7.fit(X_train, y_train_o, epochs=epAM, batch_size=bsAM, validation_data=(X_test, y_test_o))
ndf = pd.DataFrame(nn_am7.history)
# グラフ化
graph(nndf)
y_nn_am7   = modelam7.predict_classes(X_test, batch_size=bsAM, verbose=1)
print(metrics.classification_report(y_test, y_nn_am7))
cam7=metrics.confusion_matrix(y_test, y_nn_am7)
print("{}".format((cam7[0,0]+cam7[1,1])/sum(sum(cam7))))
#混同行列から、Accuracy、Recall、Precision、F1の値をDataFrameに追加する
def adddata(c,name,t_ac,cv,param):
    data = []
    Ac = (c[0,0]+c[1,1])/sum(sum(c))
    Re = c[0,0]/sum(c[0:2,0])
    Pr = c[0,0]/sum(c[0,])
    F1 = 2*(Re*Pr)/(Re+Pr)
    return pd.DataFrame({'model':[name],'Accuracy':Ac, 'Recall':Re,
                         'Precision':Pr,'F1':F1, 'train Accuracy':t_ac,
                         'cv':cv, 'param':param})
result = pd.DataFrame(index=[],columns=['model','Accuracy',
                                        'Recall','Precision','F1',
                                        'train Accuracy','cv','param'])
#重要度表示用
def importance(feature):
    pd.DataFrame(feature, index=X.columns).plot.bar(figsize=(12,3))
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.show()
#random forest評価
c = confusion_matrix(y_true=y_test, y_pred=y_forestcv,labels=[1,0])
result = result.append(adddata(c,'Random Forest CV',
                               Forestcv.score(X_train, y_train),
                               CVf,"{}".format(Forestcv.best_params_)))
importance(Forestcv.best_estimator_.feature_importances_)
#ロジスティック線形回帰評価
#liblinear
c = confusion_matrix(y_true=y_test, y_pred=y_lrLcv,labels=[1,0])
result = result.append(adddata(c,'Logistic Regression Liblinear CV', 
                               LrLcv.score(X_train, y_train),
                               CVl, "{}".format(LrLcv.best_params_)))
#Liblinear以外
c = confusion_matrix(y_true=y_test, y_pred=y_lrNLcv,labels=[1,0])
result = result.append(adddata(c,'Logistic Regression None Liblinear CV',
                               LrNLcv.score(X_train, y_train),
                               CVl, "{}".format(LrNLcv.best_params_)))
#decision tree評価
c = confusion_matrix(y_true=y_test, y_pred=y_treecv,labels=[1,0])
result = result.append(adddata(c,'Decision Tree CV',
                               Treecv.score(X_train, y_train),
                               CVt, "{}".format(Treecv.best_params_)))
#adaboost評価
c = confusion_matrix(y_true=y_test, y_pred=y_boostcv,labels=[1,0])
result = result.append(adddata(c,'adaboost CV',
                               Boostcv.score(X_train, y_train),
                               CVa,"{}".format(Boostcv.best_params_)))
#K-Neighbor評価
c = confusion_matrix(y_true=y_test, y_pred=y_kneighborcv,labels=[1,0])
result = result.append(adddata(c,'KNeighbor CV',
                               KNeighborcv.score(X_train_std, y_train),
                               CVk,"{}".format(KNeighborcv.best_params_)))
#SVM評価
c = confusion_matrix(y_true=y_test, y_pred=y_svm,labels=[1,0])
result = result.append(adddata(c,'SVM', Svm.score(X_train, y_train),0,None))

#ハイパーパラメータ
c = confusion_matrix(y_true=y_test, y_pred=y_svmcvsig,labels=[1,0])
result = result.append(adddata(c,'SVM CV Sigmoid',
                               SvmcvSig.score(X_train, y_train),
                               CVs,"{}".format(SvmcvSig.best_params_)))
c = confusion_matrix(y_true=y_test, y_pred=y_svmcvpoly,labels=[1,0])
result = result.append(adddata(c,'SVM CV Poly',
                               SvmcvPoly.score(X_train, y_train),
                               CVs,"{}".format(SvmcvPoly.best_params_)))
#NN評価
#SGD
#1層
result = result.append(adddata(c1,'NN_SGD1', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(ep,bs,lr)))
#3層
result = result.append(adddata(c3,'NN_SGD3', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(ep,bs,lr)))
#5層
result = result.append(adddata(c5,'NN_SGD5', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(ep,bs,lr)))
#7層
result = result.append(adddata(c7,'NN_SGD7', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(ep,bs,lr)))
#9層
result = result.append(adddata(c9,'NN_SGD9', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(ep,bs,lr)))
#RMS
#3層
result = result.append(adddata(cR3,'NN_RMS3', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epR,bsR,lrR)))
#5層
result = result.append(adddata(cR5,'NN_RMS5', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epR,bsR,lrR)))
#Adag
#1層
result = result.append(adddata(cag1,'NN_ADAG1', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epAG,bsAG,lrAG)))
#3層
result = result.append(adddata(cag3,'NN_ADAG3', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epAG,bsAG,lrAG)))
#Adad
#7層
result = result.append(adddata(cad7,'NN_ADAD7', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epAD,bsAD,lrAD)))
#9層
result = result.append(adddata(cad9,'NN_ADAD9', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epAD,bsAD,lrAD)))
#Adam
#3層
result = result.append(adddata(cam3,'NN_ADAM3', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epAM,bsAM,lrAM)))
#7層
result = result.append(adddata(cam7,'NN_ADAM7', None,0,'epochs={0},batchsize={1},学習率lr={2}'.format(epAM,bsAM,lrAM)))
result
print(result)
#CSV保存
result.to_csv('survey_result.csv')