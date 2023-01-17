import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
pd.options.mode.chained_assignment = None
class crypto:
    def __init__(self, df, currency = 'XLM'):
        self.currency = currency
        df = df.rename(columns={'Unnamed: 0': 'Day'})
        self.df = df[(df['Symbol']==self.currency)]
        
        del df
    
    def fit(self, target='High', type_='AR', ARIMA_order=(1,1,1)):
        """
        Fit the model on the train set
        There is a choice between AR and ARIMA models.
        The ARIMA_order parameter is only used in the latter case.
        """
       # self.df['%s_shift' % (target)] = self.df[target].shift()
        #self.df.dropna(inplace=True)
        divide=len(self.df)-200
        self.train = self.df[:divide]
        self.test = self.df[divide:]
        self.target = target
        
        if type_=='AR':
            model = AR(self.train[target])
            
        elif type_=='ARIMA':
            model = ARIMA(self.train[target], order=ARIMA_order)
        self.model_fit = model.fit()
        print('Lag: %s' % self.model_fit.k_ar)
        print('Coefficients: %s' % self.model_fit.params)
        self.window = self.model_fit.k_ar
        self.coef = self.model_fit.params
        sns.lineplot(x = 'Day', y = 'High', data=self.train, label = 'train', color='b')
        sns.lineplot(self.train['Day'], y=self.model_fit.fittedvalues,  label = 'Fitted Values', color='r')
        plt.title('Fitted Model')
        plt.show()
        
    def fit_analysis(self):
        self.train['%s_predicted' % (self.target)] = self.model_fit.fittedvalues
        return self.train
    
    def plot_rolling_mean(self, window=7):
        sns.set(rc={'figure.figsize':(20,20)})
        sns.set(style="whitegrid", font_scale=1.5)
        sns.set_palette("Paired")
        sns.lineplot(x = self.train['Day'], y = self.train[self.target].rolling(window).mean(), data=self.train, label = 'Train rolling mean', color='b')
        sns.lineplot(x=self.train['Day'], y=self.model_fit.fittedvalues.rolling(window).mean(),  label = 'Fitted Values rolling mean', color='r')
        plt.title('Rolling Mean Window = %s' % (window))
        plt.show()
        
    def ups_and_downs(self, day_shift=1, type_='train'):
        """
        Shift the target by 1 day, and calculate the difference between today and yesterday.
        Drop NaNs and assign 0 to a down-shift and 1 to an up-shift in the training set.
        Repeat this process for the predicted values.
        """
        if type_=='train':
            train = self.train.copy()
            train.dropna(inplace=True)
            train['%s_shift' % (self.target)] = train[self.target].shift(day_shift)
            train.dropna(inplace=True)
            train['%s_shift_predicted' % (self.target)] = train['%s_predicted' % (self.target)].shift(day_shift)
            train.dropna(inplace=True)
            train['Diff_%s' % (self.target)] = train['%s_shift' % (self.target)] - train[self.target]
            train['Diff_%s_predicted' % (self.target)] = train['%s_shift_predicted' % (self.target)] - train['%s_predicted' % (self.target)]
            train['Ups_Downs_Real'] = np.where(train['Diff_%s' % (self.target)]<=0, 0, 1)
            train['Ups_Downs_Predicted'] = np.where(train['Diff_%s_predicted' % (self.target)]<=0, 0, 1)
            return train
        elif type_=='test':
            test = self.test.copy()
            test.dropna(inplace=True)
            test['%s_shift' % (self.target)] = test[self.target].shift(day_shift)
            test.dropna(inplace=True)
            test['%s_shift_predicted' % (self.target)] = test['pred'].shift(day_shift)
            test.dropna(inplace=True)
            test['Diff_%s' % (self.target)] = test['%s_shift' % (self.target)] - test[self.target]
            test['Diff_%s_predicted' % (self.target)] = test['%s_shift_predicted' % (self.target)] - test['pred']
            test['Ups_Downs_Real'] = np.where(test['Diff_%s' % (self.target)]<=0, 0, 1)
            test['Ups_Downs_Predicted'] = np.where(test['Diff_%s_predicted' % (self.target)]<=0, 0, 1)
            return test
        
    def predict(self, target='High'):
        """
        Predict the future!
        """
        history = self.train[target].iloc[len(self.train)-self.window:]
        history = [history.iloc[i] for i in range(len(history))]
        predictions = list()
        for t in range(len(self.test)):
            length = len(history)
            lag = [history[i] for i in range(length-self.window,length)]
            yhat = self.coef[0]
            for d in range(self.window):
                yhat += self.coef[d+1] * lag[self.window-d-1]
            obs = self.test[target].iloc[t]
            predictions.append(yhat)
            history.append(obs)
        self.predictions = predictions
        self.test['pred'] = self.predictions
        sns.set(rc={'figure.figsize':(10,10)})
        sns.set(style="whitegrid", font_scale=1.5)
        sns.set_palette("Paired")
        sns.lineplot(x = 'Day', y = 'High', data=self.test, label = 'test', color='b')
        sns.lineplot(self.test['Day'], y=self.predictions,  label = 'predictions', color='r')
        plt.title('%s' % (self.currency))
        plt.show()
        
df = pd.read_csv("../input/all_currencies.csv")

xlm = crypto(df=df, currency='XLM')
xlm.fit(target='High')
tmp = xlm.fit_analysis()
tmp.head(25)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
tmp.dropna(inplace=True)
print('Mean Absolute Error: ', mean_absolute_error(tmp['High'], tmp['High_predicted']),'\n',
      'Mean Squared Error: ', mean_squared_error(tmp['High'], tmp['High_predicted']))
print('Mean Value of XLM: ', np.mean(tmp['High']))
xlm.plot_rolling_mean(window=7)
ups_downs = xlm.ups_and_downs(day_shift=1)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
ups_downs.head(15)
from sklearn.metrics import roc_auc_score
roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']])
ups_downs = xlm.ups_and_downs(day_shift=7)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('7 days: ', roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))

ups_downs = xlm.ups_and_downs(day_shift=14)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('14 days: ',roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))
xlm.predict()
print('Mean Absolute Error: ', mean_absolute_error(xlm.test['High'], xlm.predictions),'\n',
      'Mean Squared Error: ', mean_squared_error(xlm.test['High'], xlm.predictions))
print('Mean Value of XLM: ', np.mean(xlm.test['High']))
ups_downs = xlm.ups_and_downs(day_shift=1, type_='test')[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
ups_downs.head(15)
roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']])
ups_downs = xlm.ups_and_downs(day_shift=7, type_='test')[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('7 days: ', roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))

ups_downs = xlm.ups_and_downs(day_shift=14, type_='test')[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('14 days: ',roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))
xlm.fit(target='High', type_='ARIMA', ARIMA_order=(23,0,0))
xlm.fit(target='High', type_='ARIMA', ARIMA_order=(23,0,1))
xlm.fit_analysis()
ups_downs = xlm.ups_and_downs(day_shift=7)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']])
