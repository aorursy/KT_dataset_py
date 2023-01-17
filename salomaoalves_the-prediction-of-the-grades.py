import pandas as pd

df = pd.read_csv('/kaggle/input/enem-score/Data_Base.csv',sep=';')

df.fillna(0)

df.head()
year = [2015,2016,2017,2018]

dic = {'Year': year}

gradesMT,gradesLC = [],[]

gradesCN,gradesCH = [],[]
#average of the grades



for i in range(4):

    gradesCH.append(df[df['NU_ANO'] == year[i]].NU_NOTA_CH.mean())

    gradesCN.append(df[df['NU_ANO'] == year[i]].NU_NOTA_CN.mean())

    gradesMT.append(df[df['NU_ANO'] == year[i]].NU_NOTA_MT.mean())

    gradesLC.append(df[df['NU_ANO'] == year[i]].NU_NOTA_LC.mean())
dic['GRADES_CH'] = gradesCH

dic['GRADES_CN'] = gradesCN

dic['GRADES_MT'] = gradesMT

dic['GRADES_LC'] = gradesLC
grades = pd.DataFrame(dic)

grades.head()
from sklearn.linear_model import LinearRegression
def regLinear(x,y,x_pred):

    model = LinearRegression()

    model.fit(x,y)

    pred = []

    for i in x_pred:

        pred.append(float(model.predict([[i]]).max()))

    return pred
x = grades.iloc[:,0].values

x = x.reshape(-1,1)
y_CH = grades.iloc[:,1].values

y_CH = y_CH.reshape(-1,1)



y_CN = grades.iloc[:,2].values

y_CN = y_CN.reshape(-1,1)



y_MT = grades.iloc[:,3].values

y_MT = y_MT.reshape(-1,1)



y_LC = grades.iloc[:,4].values

y_LC = y_LC.reshape(-1,1)
pred = [2019,2020]

dic = {'Year': [2019,2020]}
dic['GRADES_CH'] = regLinear(x,y_CH,pred)

dic['GRADES_CN'] = regLinear(x,y_CN,pred)

dic['GRADES_MT'] = regLinear(x,y_MT,pred)

dic['GRADES_LC'] = regLinear(x,y_LC,pred)
grades_pred = pd.DataFrame(dic)

grades_pred.head()
final_grades = pd.concat([grades,grades_pred], ignore_index=True)

final_grades