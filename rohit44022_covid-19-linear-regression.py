#regression analysis is form if predictive modelling technique which investigaes the relation between dependent  independent variables
#firstv locakdown happen in india on 26 march 2020
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import linear_model
data = pd.read_csv("../input/covid-data/case_time_series.csv")
print(data)
X=data[['Day']]
Y=data['DailyCase']
print(X)
print(Y)
lm=linear_model.LinearRegression()
lm.fit(X,Y)

X1 = np.arange(157,300)
print(X1)
print(len(X1))

X1_array=X1.reshape(143,1)
print(X1_array)

predictions=lm.predict(X1_array)
print(predictions)

df2 = pd.DataFrame(X1_array,predictions)
print(df2)
df2.to_csv('covid_Predictionafter_200Day.csv')

X2=np.array([[44],[50],[55],[60],[65],[70],[75],[80],[85],[90],[95],[100],[105],[110],[115],[120],[125],[130],[135],[140],[145],[150]])
Y2=np.array([[3656],[4311],[3808],[5720],[6414],[8364],[9847],[9981],[11405],[14740],[16868],[18339],[24018],[25790],[29917],[40235],[48888],[52479],[50488],[65156],[64141],[65024]])
lm.score(X2,Y2)