import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import matplotlib

from glob import glob

import os



font = {'weight' : 'bold','size'   : 22}

matplotlib.rc('font', **font)



def remove():

    filelist = glob("/kaggle/working/*")

    for ifil in filelist:

        os.remove(ifil)

        

def main():

    thres = 0

    remove()

    df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv',header=0)

    dates = df.columns.values[4:]

    initd = dates[0]

    df = df.groupby('Country/Region').sum()

    df = df.T.drop(['Lat','Long'],axis=0)

    fig = plt.figure(figsize=(20,15))

    

    for i,datef in enumerate(dates):

        print(i)

        plt.title('%02.2i days since %s: %s'%(i,initd, datef))

        for icol in df.columns:

            data = df[icol].values[:i+1]

            data = data[data>thres]

            plt.plot(data)#,'.')

            if data.shape[0]>0:

                plt.text(data.shape[0], np.max(data), icol, fontsize=12)

        plt.xlim([0.,80.])

        plt.ylim(bottom=1.,top=10**5.)

        plt.yscale('log',nonposy='clip')

        plt.ylabel('Total confirmed cases')

        plt.xlabel("Days since exceedence of country's cumulative confirmed (thres = %1i)"%(thres))

        plt.savefig('%02.2i_days_since_Jan22.png'%(i),bbox_inches='tight')

        plt.clf()

        plt.cla()

        

if __name__=="__main__":

    main()