# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt





df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', index_col=0)

df.head()
# df.info()

X = df['Date']

#print(type(X))



X = pd.Series(X).values

# print(X[0])



# print(X[12095:12153])



Y = df['ConfirmedCases']

Y = pd.Series(Y).values

# print(Y[12095:12153])



plt.plot(X[3600:3660], Y[3600:3660])

# plt.axvline(x=20)
f = [Y[3600]]

R = 0.27



for d in range(1, 60):

    if (d == 6):

        R = 1.2

    elif (d == 7):

        R = 0.1

    elif (d == 8):

        R = 0.3

    elif (d == 10):

        R = 0.35

    elif (d == 12):

        R = 0.22

    elif (d == 15):

        R = 0.109

    elif (d == 19):

        R = 0.07

    elif (d == 21):

        R = 0

    elif (d == 22):

        R = 0.49

    elif (d == 23):

        R = 0.06

    elif (d == 24):

        R = 0.038

    elif (d == 27):

        R = 0

    f.append(f[d - 1] + f[d - 1] * R)

    

plt.plot(X[3600:3660], Y[3600:3660], f)

# plt.axvline(x=)
f = [Y[3600]]

R = 0.3445



for d in range(1, 60):

    if (d == 18):

        R = 0

    f.append(f[d - 1] + f[d - 1] * R)

    

plt.plot(X[3600:3660], Y[3600:3660], f)
# Taiwan: 12300:12360



f = [Y[8400]]

R = 0.615



for d in range(1, 60):

    if (d == 21):

        R = 0

    fnew = f[d - 1] + f[d - 1] * R

    f.append(fnew)

f2 = []

for d in range(0, 60):

    e = 1 - (1 / (0.1 * d + 1))

    f2.append(f[d] * e)



plt.plot(X[8400:8460], Y[8400:8460], f2)
X = df['Date']

X = pd.Series(X).values

# print(np.where(X == "Hubei"))



# Hubei: 3720:3782

# South Korea: 8680:8742



Y = df['ConfirmedCases'].values

# print(Y[8680])



# f = [Y[8680]]

f = []

R = 1



"""

for d in range(1, 60):

    if (d == 10):

        R = 0.5

    elif (d == 15):

        R = 0

    f.append(f[d - 1] + f[d - 1] * R)

"""

count = 0

inds = [0]



for d in range(8680, 8742):

    f.append(Y[d])

    ind = f.index(Y[d])

    if ((f[ind] - f[ind - 1]) > 10 * (count + 3)):

        count = count+1

        # print(ind, "!!!!!")

        inds.append(ind - 1)

        # plt.axvline(x=X[ind])



# print(inds)

inds.append(len(f) - 1)



slopes = []

for i in range(1, len(inds)):

    # print(inds[i])

    # print("f[i] - f[i - 1]:", f[inds[i]], f[inds[i - 1]])

    print("range:", inds[i], inds[i - 1])

    slope = (f[inds[i]] - f[inds[i - 1]] ) / (inds[i] - inds[i - 1])

    print("slope:", slope)

    print()

    slopes.append(slope)

    plt.axvline(x = inds[i])



"""

print("!!!!!")



print("last day:", len(f) - 1)

print("cases:", f[len(f) - 1])

print("last index:", inds[len(inds) - 1])

print("cases:", f[inds[len(inds) - 1]])



print("!!!!!")

"""



print(slopes)



plt.plot(X[8680:8742], Y[8680:8742], f)
# print(slopes)

# print(inds[0])



for s in range(1, len(slopes)):

    # print("slope:", slopes[s - 1])

    # print("days:", inds[s - 1], "to", inds[s])

    # print()

    

    if (slopes[s - 1] > slopes[s]):

        # print("!!!")

        # print("slope yesterday and today:", slopes[s - 1], slopes[s])

        # print("DAYS WITH IMPROVEMENT:", inds[s], "to", inds[s + 1])

        plt.axvspan(inds[s], inds[s+1], alpha=0.5, color='red')



        

# print("!!!!!!!!!!!!")



# plt.axvline(x = 30)

plt.plot(X[8680:8742], Y[8680:8742])
start = 8680

end = 8742

count = end - start



dy = np.diff(Y[start:end])

dy = np.append(0, dy)



plt.plot(X[start:end], Y[start:end])

# plt.plot(X[start:end], dy)



cshift = 2



dys = np.convolve(dy, [0.2, 0.2, 0.2, 0.2, 0.2])

# dys = [0, 0, 0].append(dys)

# print(type(dys));

plt.plot(X[start:end], dys[cshift: -cshift])



inds = np.where(dys == dys.max())

# print(inds)

maxDate = np.mean(inds) - cshift

# print(maxDate)

plt.axvline(x = maxDate,  color = 'red')
C = df['Province/State']

C = pd.Series(C).values

# print(np.where(C == "Anhui"))



X = df['Date'].values

Y = df['ConfirmedCases'].values



start = 2914

end = 2976



plt.plot(X[start:end], Y[start:end])



count = end - start



dy = np.diff(Y[start:end])

dy = np.append(0, dy)



plt.plot(dy)



dys = np.convolve(dy, [0.2, 0.2, 0.2, 0.2, 0.2])



plt.plot(dys[cshift:-cshift])



inds = np.where(dys == dys.max())

# print(inds)

maxDate = np.mean(inds) - cshift

# print(maxDate)

plt.axvline(x = maxDate,  color = 'red')
C = df['Province/State']

C = pd.Series(C).values

i = np.where(C == "Anhui")



X = df['Date'].values

Y = df['ConfirmedCases'].values



start = i[0][0]

end = i[0][len(i) - 2] + 1



plt.plot(X[start:end], Y[start:end])



dy = np.append(0, np.diff(Y[start:end]))



dys = np.convolve(dy, [0.2, 0.2, 0.2, 0.2, 0.2])



plt.plot(dys[cshift:-cshift])



inds = np.where(dys == dys.max())

maxDate = np.mean(inds) - cshift

print("Day Improved:", maxDate)

plt.axvline(x = maxDate,  color = 'red')
C = df['Province/State']

C = pd.Series(C).values

i = np.where(C == "Chongqing")



X = df['Date'].values

Y = df['ConfirmedCases'].values



start = i[0][0]

end = i[0][len(i) - 2] + 1



plt.plot(X[start:end], Y[start:end])



dy = np.append(0, np.diff(Y[start:end]))



dys = np.convolve(dy, [0.2, 0.2, 0.2, 0.2, 0.2])



plt.plot(dys[cshift:-cshift])



inds = np.where(dys == dys.max())

maxDate = np.mean(inds) - cshift

print("Day Improved:", maxDate)

plt.axvline(x = maxDate,  color = 'red')
# f = [Y[int(maxDate)]]

f = [Y[start]]



R = 0.01



# while (max(f) <= max(Y[start:end])):

    

f = [Y[start]]

R = R + 0.04



for d in range(1, 60):



    if (d >= maxDate):

        if (d == 30):

            R = 0

        f.append(f[d - 1] + f[d - 1] * R)

    else:

        f.append(Y[d + start])

    

    # R = R + 0.01



# print(R)

plt.plot(X[start:end], Y[start:end], dys)

plt.plot(f)

# print(maxDate)

plt.axvline(x=14)

print(max(f) <= max(Y[start:end]))
C = df['Province/State']

C = pd.Series(C).values

i = np.where(C == "Hainan")



X = df['Date'].values

Y = df['ConfirmedCases'].values



start = i[0][0]

end = i[0][len(i) - 2] + 1



plt.plot(X[start:end], Y[start:end])



dy = np.append(0, np.diff(Y[start:end]))



dys = np.convolve(dy, [0.2, 0.2, 0.2, 0.2, 0.2])



plt.plot(dys[cshift:-cshift])



inds = np.where(dys == dys.max())

maxDate = np.mean(inds) - cshift

print("Day Improved:", maxDate)

plt.axvline(x = maxDate,  color = 'red')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv') #, index_col=0

df.head()

DX = df['Date'].values

DP = df['Province/State'].values

DC = df['Country/Region'].values

DY = df['ConfirmedCases'].values

DZ = df['Fatalities'].values
dt = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv') #, index_col=0

#dt.head()



DTX = dt['Date'].values

DTP = dt['Province/State'].values

DTC = dt['Country/Region'].values

def get_status():

    #plt.plot(DX[start:end], Y)

    # derivative

    Y2 = np.diff(Y)

    Y2 = np.append(0, Y2)

    #plt.plot(X[start:end], Y2)

    # smooth

    conv_pad = 2

    Y3 = np.convolve(Y2, [0.2, 0.2, 0.2, 0.2, 0.2], mode='valid')

    Y4 = np.append([float('NaN'), float('NaN')], Y3)

    Y4 = np.append(Y4, [float('NaN'), float('NaN')])

    #plt.plot(X[start:end], Y4)

    # max

    max_change = Y3.max()

    idx = np.where(Y3==max_change)

    max_date = int(np.mean(idx)) + conv_pad

    #print(max_date)

    status = 0

    if (max_date >= count - conv_pad - 2):

        # has not peaked yet

        if max_change > 0:

            # rising

            status = 1

        else:

            # flat

            status = 0

    else:

        # has peaked

        status = 2

        

    return (status, max_date)
def get_params(iid, oid, ofield):

    print(iid, oid, ofield)

    offset = 50

    remain = count - max_date

    if status == 2:

        # peaked: extrapolate low R

        if remain > 10:

            remain = 10

        start2 = count - remain

        R_reduction = 0.95



        # find best-fit R

        loops = 50

        R0 = 0

        step = 0.1

        prev_err = float('inf')

        prev_R0 = R0

        prev2_R0 = R0

        while loops > 0:

            err = 0

            R = R0

            F = [Y[start2]]

            for d in range(1,remain):

                fc = F[d-1] + F[d-1]*R

                F.append(fc)

                R = R * R_reduction

                err += abs(fc - Y[start2+d])

            #print(R0, err)

            if (abs(err - prev_err) / (err + 0.01) < 0.01):

                break

            if (err > prev_err):

                # getting worse

                step /= 10

                prev_R0 = prev2_R0

                R0 = prev2_R0 + step

                prev_err = float('inf')

            else:

                prev_err = err

                prev2_R0 = prev_R0

                prev_R0 = R0

                R0 += step



            loops -= 1



        R = R0

        F = [Y[start2]]

        oi = oid

        dt.loc[dt.ForecastId == oi, ofield] = Y[offset]

        for d in range(1, ocount): # starting at offset

            

            if d <= start2 - offset:

                dt.loc[dt.ForecastId == oi, ofield] = Y[d + offset]

            else:

                ii = d + offset - start2 # starts at 1

                fc = F[ii-1] + F[ii-1]*R

                F.append(fc)

                if d < remain:

                    dt.loc[dt.ForecastId == oi, ofield] = Y[d + offset]

                else:

                    dt.loc[dt.ForecastId == oi, ofield] = float(int(fc))

                    

            R = R * R_reduction

            oi += 1



    else:

        # rising: extrapolate low R

        remain = 10

        start2 = count - remain

        R_reduction = 0.8

        days_to_peak = 7



        # find best-fit R

        loops = 50

        R0 = 0

        step = 0.1

        prev_err = float('inf')

        prev_R0 = R0

        prev2_R0 = R0

        while loops > 0:

            err = 0

            R = R0

            F = [Y[start2]]

            for d in range(1,remain):

                fc = F[d-1] + F[d-1]*R

                F.append(fc)

                if d > days_to_peak:

                    R = R * R_reduction

                err += abs(fc - Y[start2+d])

            #print(R0, err)

            if (abs(err - prev_err) / (err + 0.01) < 0.01):

                break

            if (err > prev_err):

                # getting worse

                step /= 10

                prev_R0 = prev2_R0

                R0 = prev2_R0 + step

                prev_err = float('inf')

            else:

                prev_err = err

                prev2_R0 = prev_R0

                prev_R0 = R0

                R0 += step



            loops -= 1



        R = R0

        F = [Y[start2]]

        oi = oid

        dt.loc[dt.ForecastId == oi, ofield] = Y[offset]

        oi += 1

        for d in range(1, ocount): # starting at offset

            

            if d <= start2 - offset:

                dt.loc[dt.ForecastId == oi, ofield] = Y[d + offset]

                #print(oi, Y[d + offset])

            else:

                ii = d + offset - start2 # starts at 1

                fc = F[ii-1] + F[ii-1]*R

                F.append(fc)

                if d < remain:

                    dt.loc[dt.ForecastId == oi, ofield] = Y[d + offset]

                    #print(oi, Y[d + offset])

                else:

                    dt.loc[dt.ForecastId == oi, ofield] = float(int(fc))

                    #print(oi, float(int(fc)))

                    

            if d > days_to_peak:

                R = R * R_reduction

            oi += 1

            

    #time = list(range(end - start))

    #plt.plot(time, Y)

    #plt.plot(list(range(start2, start2 + len(F))), F)            



    return (R0, start2)
# main loop



rows = len(df)



start = 0



i = 0

while True:

    province = df.iloc[start, 1]

    country = df.iloc[start, 2]

    if isinstance(province, str):

        idx = np.where((DC == country) & (DP == province))

    else:

        idx = np.where(DC == country)

    start = np.min(idx)

    end = np.max(idx) + 1

    count = end - start



    # output

    if isinstance(province, str):

        idx = np.where((DTC == country) & (DTP == province))

    else:

        idx = np.where(DTC == country)

    ostart = np.min(idx)

    oend = np.max(idx) + 1

    ocount = oend - ostart



    if isinstance(province, str):

        key = province + '_' + country

    else:

        key = country



    # confirmed cases =======================

    Y = DY[start:end]

    

    (status, max_date) = get_status()

    print(status)

    

    # do work here

    (Rinit, start2) = get_params(df.iloc[start, 0], dt.iloc[ostart, 0], 'Lat')

    print('params["' + key + '"] = (' + str(status) + ', ' + str(start2) + ', ' + str(Rinit) + ')')



    # fatalities =======================

    Y = DZ[start:end]

    

    (status, max_date) = get_status()

    

    # do work here

    (Rinit, start2) = get_params(df.iloc[start, 0], dt.iloc[ostart, 0], 'Long')

    print('params["' + key + '"] = (' + str(status) + ', ' + str(start2) + ', ' + str(Rinit) + ')')



    # end =======================

    start = end

    if (start >= rows):

        break

    i += 1
dt.to_csv('submission.csv', index = False, columns = ['ForecastId', 'Lat', 'Long'], header = ['ForecastId', 'ConfirmedCases', 'Fatalities'])

ds = pd.read_csv('submission.csv')

ds.head()
ds[0:50]