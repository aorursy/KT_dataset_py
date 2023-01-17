from pandas import DataFrame 
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as kuvaaja

data = {'työvuosia_takana': [1.01, 1.03, 1.05, 2.00, 2.02, 2.09, 3.00, 3.02, 3.02, 3.07, 3.09, 4.00, 4.00, 4.01, 4.05, 4.09, 5.01, 5.03, 5.09, 6.00, 6.08, 7.01, 7.09, 8.02, 8.07, 9.00, 9.05, 9.06, 10.03, 10.05],
        'palkan_suuruus': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
             }
data_frame = DataFrame(data)
data_frame.head(30)
kuvaaja.title('palkan suuruus vs tyovuoisa takana', fontsize=14)
kuvaaja.xlabel('työvuosia_takana', fontsize=14)
kuvaaja.ylabel('palkan_suuruus', fontsize=14)
kuvaaja.grid(True) 
kuvaaja.scatter(data_frame['työvuosia_takana'], data_frame['palkan_suuruus'])
kuvaaja.show()



kuvaaja.title('palkan suuruus vs tyovuoisa takana', fontsize=14)
kuvaaja.xlabel('työvuosia_takana', fontsize=14)
kuvaaja.ylabel('palkan_suuruus', fontsize=14)
kuvaaja.grid(True) 
kuvaaja.scatter(data_frame['työvuosia_takana'], data_frame['palkan_suuruus'])

regressiotyokalu = linear_model.LinearRegression()
regressiotyokalu.fit(data_frame[['työvuosia_takana']], data_frame['palkan_suuruus'])

regressioviiva = regressiotyokalu.predict(data_frame[['työvuosia_takana']])
kuvaaja.plot(data_frame['työvuosia_takana'], regressioviiva, color='y')
kuvaaja.show()

print('Leikkauspiste: \n', regressiotyokalu.intercept_)

print('Kertoimet: \n', regressiotyokalu.coef_)

regressiotyokalu.predict(np.array([4,20,16]).reshape(-1, 1))

