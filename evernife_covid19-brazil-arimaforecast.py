import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.statespace.mlemodel import MLEResults

import datetime
#Ultimos 10 dias  de morte no BRASIL!

original_df = pd.read_csv("../input/covid19-brazil-08052020/cases-brazil-states-08-05-2020.csv");

df_br = original_df.filter(['date', 'state', 'deaths']);

df_br = df_br.loc[df_br['state'] == "TOTAL"];



print("Ultimos 10 dias de mortes no Brasil")

print(df_br.tail(10));

print("\n\nGráfico da curva de crescimento das mortes")



fig = px.bar(df_br, x="date", y="deaths", color="deaths", barmode="group")

fig.show()
df = original_df.loc[original_df['date'] == "2020-05-08"];

df = df.loc[df['state'] != "TOTAL"];

df = df.loc[df['deaths'] > 100];

df = df.filter(['state', 'deaths']);

df = df.sort_values(by=['deaths'],ascending=False)



print("\n\nRelação de mortes por estados com mais de 100 mortes")

fig = px.bar(df, x="state", y="deaths", color="state", barmode="group", width=1000)

fig.show()
#Ultimos 10 dias  de morte no Estado DE SÃO PAULO!

df_sp = original_df.filter(['date', 'state', 'deaths']);

df_sp = df_sp.loc[df_sp['state'] == "SP"];



print("Ultimos 10 dias de Mortes no Estado de São Paulo")

print(df_sp.tail(10));



print("\n\nGráfico da curva de crescimento das mortes")



#foundNull = df_sp['deaths'].isnull().values.any(); #Nenhum valor nulo encontrado





fig = px.bar(df_sp, x="date", y="deaths", color="deaths", barmode="group")

fig.show()
class AnalyticsARIMA():



    # ---------------------------------------------------------------------------

    #   Arima

    # ---------------------------------------------------------------------------



    modelFit:MLEResults = None;

    nomeDaColunaObjetivo = None;

    nomeDaColunaDeDatas = None;



    def arimaDefinirColunaObjetivo(self, nomeDaColunaObjetivo, nomeDaColunaDeDatas, funcaoDeConversaDeDatas=None):

        self.nomeDaColunaObjetivo = nomeDaColunaObjetivo;

        self.nomeDaColunaDeDatas = nomeDaColunaDeDatas;



        if funcaoDeConversaDeDatas is not None:

            self.df[nomeDaColunaDeDatas] = self.df[nomeDaColunaDeDatas].apply(funcaoDeConversaDeDatas);

        else:

            self.df[nomeDaColunaDeDatas] = pd.to_datetime(self.df[nomeDaColunaDeDatas])



        self.df = self.df.groupby(nomeDaColunaDeDatas)[nomeDaColunaObjetivo].sum().reset_index()

        self.df = self.df.set_index(nomeDaColunaDeDatas)



    #Atenção, o dataFrame precisa estar AGRUPADO e ORDENADO

    def plotarDecomposicao(self, dataFrame=None, theModel='addtive', theFigsize = None, theFreq = None):

        from pylab import rcParams;

        import statsmodels.api as sm;

        rcParams['figure.figsize'] = 18, 8;



        if dataFrame is None:

            dataFrame = self.df;



        # Dois tipos de modelos possíves, Aditivo e Multiplicativo (Necessário testar as diferenças)

        if theFreq is not None:

            decomposicao = sm.tsa.seasonal_decompose(dataFrame, model=theModel, freq=theFreq);

        else:

            decomposicao = sm.tsa.seasonal_decompose(dataFrame, model=theModel);

        if theFigsize is not None:

            plt.figure(figsize=theFigsize)

        decomposicao.plot();





    #ARIMA_SASONALIDADE == 12 meses, no caso, 1 ano;

    def aplicarARIMA(self, ARIMA_SASONALIDADE = 12, verbose = False):



        if verbose:

            print('# ===============================================================');

            print('# Preparando quantidade de treino.');

            print('# ===============================================================');



        import itertools;

        p = sazonalidade = range(0, 2);  # Arima P == auto-regressive part of the model

        d = tendencia = range(0, 2);  # Arima D == integrated part of the model

        q = ruido = range(0, 2);  # Arima Q == moving average part of the model



        # itertools.product basicamente relaciona todas as variáveis com todas as varíaveis... como já diz, PRODUCT

        pdq = list(itertools.product(sazonalidade, tendencia, ruido));



        # Criando agora as variações de calculos para o arima usar.

        # (Similar ao 'grid search' de machine learning)

        seasonal_pdq = [(x[0], x[1], x[2], ARIMA_SASONALIDADE) for x in list(itertools.product(p, d, q))];



        if verbose == True:

            print(seasonal_pdq);



        if verbose:

            print('# ===============================================================');

            print('# Escolhendo a melhor combinação de parametros arima.');

            print('# ===============================================================');



        import warnings;

        warnings.filterwarnings("ignore")  # Negócio chato pacas...



        menorCombinacao = None;

        menorCombinacaoValor = 99999999999999999;  # Mesma coisa que Integer.MAX_VALUE



        import statsmodels.api as sm;



        for parametro in pdq:

            for parametro_sasonal in seasonal_pdq:

                try:

                    mod = sm.tsa.statespace.SARIMAX(self.df,

                                                    order=parametro,

                                                    seasonal_order=parametro_sasonal,

                                                    enforce_stationarity=False,

                                                    enforce_invertibility=False)



                    resultado = mod.fit(disp=0)#disp == 0 Oculta log indesejado que trava o programa....



                    if resultado.aic < menorCombinacaoValor:

                        menorCombinacao = [parametro, parametro_sasonal, ARIMA_SASONALIDADE];

                        menorCombinacaoValor = resultado.aic;



                    if verbose == True:

                        print('ARIMA{}x{}x{} - AIC:{}'.format(parametro, parametro_sasonal, ARIMA_SASONALIDADE, resultado.aic))



                except:

                    # Algumas combinações são NaN (Não são possíveis! por isso tem esse TryCath)

                    continue



        if verbose == True:

            print('\n\n')

            print('O menor valor encontrado para o AIC é: {}'.format(menorCombinacaoValor))

            print('Utilizando a combinação: ARIMA{}x{}x{}'.format(menorCombinacao[0], menorCombinacao[1], menorCombinacao[2]))



        theOrder = menorCombinacao[0];

        theSeasonal_order = menorCombinacao[1];



        if verbose:

            print('# ===============================================================');

            print('# Ajustando Modelo.');

            print('# ===============================================================');



        mod = sm.tsa.statespace.SARIMAX(self.df,

                                        order=theOrder,

                                        seasonal_order=theSeasonal_order,

                                        enforce_stationarity=False,

                                        enforce_invertibility=False)



        self.modelFit = mod.fit(disp=0)#disp == 0 Oculta log indesejado que trava o programa....

        print('\n\n');

        print(self.modelFit.summary().tables[1])



    def diagnostico(self):

        self.modelFit.plot_diagnostics(figsize=(15, 12))

        plt.show()



    def ARIMAPrediction(self, forecastStartingDate = None ,datasetStartDate = None, theFigsize = (14, 7)):



        if datasetStartDate == None:

            datasetStartDate = self.df.index[0];



        if forecastStartingDate == None:

            forecastStartingDate = self.df.index[0];



        # Predição propriamente dita

        pred = self.modelFit.get_prediction(start=pd.to_datetime(forecastStartingDate), dynamic=False)

        pred_ci = pred.conf_int()



        ax = self.df[datasetStartDate:].plot(label='Observado')

        pred.predicted_mean.plot(ax=ax, label='Predicted', alpha=.7, figsize=theFigsize)





        ax.fill_between(pred_ci.index,

                        pred_ci.iloc[:, 0],

                        pred_ci.iloc[:, 1], color='k', alpha=.2)



        ax.set_xlabel(self.nomeDaColunaDeDatas)

        ax.set_ylabel(self.nomeDaColunaObjetivo)

        plt.legend()

        plt.show()



    def ARIMAForecast(self, steps, datasetStartDate = None, theFigsize = (14, 7), verbose = False):



        if datasetStartDate == None:

            datasetStartDate = self.df.index[0];



        pred_uc = self.modelFit.get_forecast(steps=steps)



        if verbose is True:

            print(pred_uc.predicted_mean)



        pred_ci = pred_uc.conf_int()



        ax = self.df[datasetStartDate:].plot(label='Observado')

        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=theFigsize)



        ax.fill_between(pred_ci.index,

                        pred_ci.iloc[:, 0],

                        pred_ci.iloc[:, 1], color='k', alpha=.2)



        ax.set_xlabel(self.nomeDaColunaDeDatas)

        ax.set_ylabel(self.nomeDaColunaObjetivo)

        plt.legend()

        plt.show()



    def ARIMAPredictionToPred(self, forecastStartingDate = None ,datasetStartDate = None):



        if datasetStartDate == None:

            datasetStartDate = self.df.index[0];



        if forecastStartingDate == None:

            forecastStartingDate = self.df.index[0];



        # Predição propriamente dita

        pred = self.modelFit.get_prediction(start=pd.to_datetime(forecastStartingDate), dynamic=False)

        return pred;



    def ARIMAForecastToPred(self, steps, datasetStartDate = None, verbose = False):



        if datasetStartDate == None:

            datasetStartDate = self.df.index[0];



        pred = self.modelFit.get_forecast(steps=steps)



        if verbose is True:

            print(pred.predicted_mean)



        return pred;



    def correlacao(self):

        plot_acf(self.df,lags=50,title="Autocorrelação")



    def correlacao_parcial(self):

        plot_pacf(self.df,lags=50,title="Autocorrelação Parcial")
print("Aplicando ARIMA para os dados nacionais")

analytics_br = AnalyticsARIMA();

analytics_br.df = df_br;

analytics_br.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='deaths', nomeDaColunaDeDatas='date')

analytics_br.aplicarARIMA(verbose=True);
analytics_br.plotarDecomposicao()
analytics_br.correlacao();
print("Aplicando ARIMA para os dados do Estado de São Paulo")

analytics_sp = AnalyticsARIMA();

analytics_sp.df = df_sp;

analytics_sp.arimaDefinirColunaObjetivo(nomeDaColunaObjetivo='deaths', nomeDaColunaDeDatas='date')

analytics_sp.aplicarARIMA(verbose=True);
analytics_sp.plotarDecomposicao()
analytics_sp.correlacao();
analytics_br.ARIMAPrediction("25-04-2020")
analytics_br.ARIMAForecast(steps=10)
df = analytics_br.df;

df_ultimos_10 = df.tail(10);

analytics_br.df = df_ultimos_10;

analytics_br.ARIMAForecast(steps=10)

analytics_br.df = df; #Voltando dataset ao normal
pred = analytics_br.ARIMAForecastToPred(steps=10)

pred.conf_int()
import plotly.graph_objects as go

from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(

    go.Scatter(x=df_ultimos_10.index, y=df_ultimos_10['deaths'], name="Real"),

    secondary_y=True,

)

fig.add_trace(

    go.Scatter(x=pred.predicted_mean.index, y=pred.predicted_mean, name="Predição"),

    secondary_y=True,

)

fig.update_layout(

    title_text="Forecast Brasil 08-05-2020 - 10 steps ahead"

)

fig.update_xaxes(title_text="Data")

fig.update_yaxes(title_text="<b>Real</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>Predição</b>", secondary_y=True)

fig.show()
analytics_sp.ARIMAPrediction("25-04-2020")
analytics_sp.ARIMAForecast(steps=10)
df = analytics_sp.df;

df_ultimos_10 = df.tail(10);

analytics_sp.df = df_ultimos_10;

analytics_sp.ARIMAForecast(steps=10)

analytics_sp.df = df; #Voltando dataset ao normal
pred = analytics_sp.ARIMAForecastToPred(steps=10)

pred.conf_int()
import plotly.graph_objects as go

from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(

    go.Scatter(x=df_ultimos_10.index, y=df_ultimos_10['deaths'], name="Real"),

    secondary_y=True,

)

fig.add_trace(

    go.Scatter(x=pred.predicted_mean.index, y=pred.predicted_mean, name="Predição"),

    secondary_y=True,

)

fig.update_layout(

    title_text="Forecast Estado de São Paulo 08-05-2020 - 10 steps ahead"

)

fig.update_xaxes(title_text="Data")

fig.update_yaxes(title_text="<b>Real</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>Predição</b>", secondary_y=True)

fig.show()


