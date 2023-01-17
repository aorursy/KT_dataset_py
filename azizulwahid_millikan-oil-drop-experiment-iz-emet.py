import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit as fit
from skimage import io
%matplotlib inline
drop1=pd.read_csv('../input/milikanfinal/p1 - Sheet1.csv')
drop2=pd.read_csv('../input/milikanfinal/p2 - Sheet1.csv')
drop3=pd.read_csv('../input/milikanfinal/p3 - Sheet1.csv')
drop4=pd.read_csv('../input/milikanfinal/p6-Sheet1.csv')
drop5=pd.read_csv('../input/milikanfinal/p7-Sheet1.csv')
drop6=pd.read_csv('../input/milikanfinal/p8-Sheet1.csv')

drop1.head()
peak1 = signal.find_peaks_cwt(drop1['x'], np.arange(1,700)) #Finding peaks
val1 = signal.find_peaks_cwt(-drop1['x'], np.arange(1, 700)) #Finding valleys

peak1 = np.delete(peak1, [0, peak1.size-1]) # Deleting false positives
val1 = np.delete(val1, 0)

f, ax = plt.subplots(figsize=(10, 5)) #Plotting journey along with peak and valley points
ax.scatter(drop1['frame'], drop1['x'])
ax.scatter(peak1, [drop1.iloc[i]['x'] for i in peak1])
ax.scatter(val1, [drop1.iloc[i]['x'] for i in val1])
ax.set(title='Rise and fall data for Droplet 1', xlabel='Frame number', ylabel='Pixel coordinate')
plt.show()
rises1 = [np.arange(peak1[i], val1[i]) for i in range(10)]
falls1 = [np.arange(val1[i], peak1[i+1]) for i in range(9)]

f, ax = plt.subplots(figsize=(15, 5))
num=1 # Pick a number in the range 1-10

ax.plot([drop1.iloc[i]['frame'] for i in rises1[num]], 
        [drop1.iloc[i]['x'] for i in rises1[num]])

ax.plot([drop1.iloc[i]['frame'] for i in falls1[num]], 
        [drop1.iloc[i]['x'] for i in falls1[num]])

plt.show()
def linefit(df, rang): 
    popt, pcov = fit(lambda x, a, b: a*x+b, [df.iloc[i]['frame'] for i in rang], [df.iloc[i]['x'] for i in rang])
    return popt[0]
rises1[1].shape
vup1 = [linefit(drop1, rises1[i]) for i in range(len(rises1))]
vdown1 = [linefit(drop1, falls1[i]) for i in range(len(falls1))]

f, ax = plt.subplots(figsize = (15, 5))
ax.bar(np.arange(len(vup1)), vup1)
ax.bar(np.arange(len(vdown1)), vdown1)
ax.set(title = 'Velocity of Rises and Falls', xticks=np.arange(10), ylabel='Velocity (Pixels/Frame)')
plt.show()
peak2 = signal.find_peaks_cwt(drop2['x'], np.arange(1, 700))
val2 = signal.find_peaks_cwt(-drop2['x'], np.arange(1, 700))

# Getting rid of outliers
peak2 = np.delete(peak2, [5, 7, 8, peak2.size-1])
val2 = np.delete(val2, [5, 7])
f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(drop2['frame'], drop2['x'])
ax.scatter(peak2, [drop2.iloc[i]['x'] for i in peak2])
ax.scatter(val2, [drop2.iloc[i]['x'] for i in val2])
ax.set(title='Positional data in vertical direction', 
       ylabel='Position in vertical direction in pixels', 
       xlabel='Frame')
plt.show()
rises2 = [np.arange(peak2[i], val2[i]) for i in range(len(peak2))]
falls2 = [np.arange(val2[i], peak2[i+1]) for i in range(len(peak2)-1)]

f, ax = plt.subplots(figsize=(15, 5))
num = 2 #Max num is 32

ax.plot([drop2.iloc[i]['frame'] for i in rises2[num]], [drop2.iloc[i]['x'] for i in rises2[num]])
ax.plot([drop2.iloc[i]['frame'] for i in falls2[num]], [drop2.iloc[i]['x'] for i in falls2[num]])
plt.show()
def linefit(df, rang): 
    popt, pcov = fit(lambda x, a, b: a*x+b, [df.iloc[i]['frame'] for i in rang], [df.iloc[i]['x'] for i in rang])
    return popt[0], np.sqrt(pcov[0,0]*pcov[0,0])
vup2 = [linefit(drop2, rises2[i])[0] for i in range(len(rises2))]
vuperr2 = [linefit(drop2, rises2[i])[1] for i in range(len(rises2))]
vdown2 = [linefit(drop2, falls2[i])[0] for i in range(len(falls2))]
vdownerr2 = [linefit(drop2, falls2[i])[1] for i in range(len(falls2))]
f, ax = plt.subplots(figsize = (15, 10))
ax.bar(np.arange(len(vup2)), vup2, yerr = vuperr2)
ax.bar(np.arange(len(vdown2)), vdown2, yerr = vdownerr2)
ax.set(ylabel='Velocity (Pixels/Frame)', xticks=np.arange(len(vup2)))
ax.legend(['Rises', 'Falls'])
plt.show()
peak3a = signal.find_peaks_cwt(drop3['x'], np.arange(1, 700))
val3a = signal.find_peaks_cwt(-drop3['x'], np.arange(1, 700))
#The 'a' at the end is to differentiate these peaks and valleys from the
#corrected peaks and valleys which I will create later, which have the 
#outliers removed. This way if I run that code again it wont remove values
#that I want to keep.
f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(drop3['frame'], drop3['x'])
ax.scatter(peak3a, [drop3.iloc[i]['x'] for i in peak3a])
ax.scatter(val3a, [drop3.iloc[i]['x'] for i in val3a])

for i in range(len(peak3a)): 
    ax.annotate('{0}'.format(i), xy=(peak3a[i], drop3.iloc[peak3a[i]]['x']))
for i in range(len(val3a)): 
    ax.annotate('{0}'.format(i), xy=(val3a[i], drop3.iloc[val3a[i]]['x']))
peak3 = np.delete(peak3a, [3, len(peak3a)-1])
val3 = np.delete(val3a, [3])

f, ax = plt.subplots(figsize = (15, 5))
ax.scatter(drop3['frame'], drop3['x'])
ax.scatter(peak3, [drop3.iloc[i]['x'] for i in peak3])
ax.scatter(val3, [drop3.iloc[i]['x'] for i in val3])

for i in range(len(peak3)): 
    ax.annotate('{0}'.format(i), xy=(peak3[i], drop3.iloc[peak3[i]]['x']))
for i in range(len(val3)): 
    ax.annotate('{0}'.format(i), xy=(val3[i], drop3.iloc[val3[i]]['x']))
rises3 = [np.arange(peak3[i], val3[i]) for i in range(len(peak3))]
falls3 = [np.arange(val3[i], peak3[i+1]) for i in range(len(peak3)-1)]

f, ax = plt.subplots(figsize=(15, 5))
num= 1 # Change to view different rises/falls
ax.plot([drop3.iloc[i]['frame'] for i in rises3[num]], [drop3.iloc[i]['x'] for i in rises3[num]])
ax.plot([drop3.iloc[i]['frame'] for i in falls3[num]], [drop3.iloc[i]['x'] for i in falls3[num]])
plt.show()
vup3 = [linefit(drop3, rises3[i])[0] for i in range(len(rises3))]
vdown3 = [linefit(drop3, falls3[i])[0] for i in range(len(falls3))]
f, ax = plt.subplots(figsize = (15, 10))
ax.bar(np.arange(len(vup3)), vup3)
ax.bar(np.arange(len(vdown3)), vdown3)
ax.set(xticks=np.arange(len(vup3)))
plt.show()
peak4 = signal.find_peaks_cwt(drop4['x'], np.arange(1,700)) #Finding peaks
val4 = signal.find_peaks_cwt(-drop4['x'], np.arange(1, 700)) #Finding valleys

peak4 = np.delete(peak4, [0, peak4.size-1]) # Deleting false positives
val4 = np.delete(val4, 0)

f, ax = plt.subplots(figsize=(10, 5)) #Plotting journey along with peak and valley points
ax.scatter(drop4['frame'], drop4['x'])
ax.scatter(peak4, [drop4.iloc[i]['x'] for i in peak4])
ax.scatter(val4, [drop4.iloc[i]['x'] for i in val4])
ax.set(title='Rise and fall data for Droplet 4', xlabel='Frame number', ylabel='Pixel coordinate')
plt.show()
rises4 = [np.arange(peak4[i], val4[i]) for i in range(10)]
falls4 = [np.arange(val4[i], peak4[i+1]) for i in range(9)]

f, ax = plt.subplots(figsize=(15, 5))
num=1 # Pick a number in the range 1-10

ax.plot([drop4.iloc[i]['frame'] for i in rises4[num]], 
        [drop4.iloc[i]['x'] for i in rises4[num]])

ax.plot([drop4.iloc[i]['frame'] for i in falls4[num]], 
        [drop4.iloc[i]['x'] for i in falls4[num]])

plt.show()
def linefit(df, rang): 
    popt, pcov = fit(lambda x, a, b: a*x+b, [df.iloc[i]['frame'] for i in rang], [df.iloc[i]['x'] for i in rang])
    return popt[0], np.sqrt(pcov[0,0]*pcov[0,0])
vup4 = [linefit(drop4, rises4[i])[0] for i in range(len(rises4))]
vuperr4 = [linefit(drop4, rises4[i])[1] for i in range(len(rises4))]
vdown4 = [linefit(drop4, falls4[i])[0] for i in range(len(falls4))]
vdownerr4 = [linefit(drop4, falls4[i])[1] for i in range(len(falls4))]
f, ax = plt.subplots(figsize = (15, 10))
ax.bar(np.arange(len(vup4)), vup4, yerr = vuperr4)
ax.bar(np.arange(len(vdown4)), vdown4, yerr = vdownerr4)
ax.set(ylabel='Velocity (Pixels/Frame)', xticks=np.arange(len(vup4)))
ax.legend(['Rises', 'Falls'])
plt.show()
peak5 = signal.find_peaks_cwt(drop5['x'], np.arange(1,700)) #Finding peaks
val5 = signal.find_peaks_cwt(-drop5['x'], np.arange(1, 700)) #Finding valleys

peak5 = np.delete(peak5, [0, peak5.size-1]) # Deleting false positives
val5 = np.delete(val5, 0)

f, ax = plt.subplots(figsize=(10, 5)) #Plotting journey along with peak and valley points
ax.scatter(drop5['frame'], drop5['x'])
ax.scatter(peak5, [drop5.iloc[i]['x'] for i in peak5])
ax.scatter(val5, [drop5.iloc[i]['x'] for i in val5])
ax.set(title='Rise and fall data for Droplet 5', xlabel='Frame number', ylabel='Pixel coordinate')
plt.show()
rises5 = [np.arange(peak5[i], val5[i]) for i in range(10)]
falls5 = [np.arange(val5[i], peak5[i+1]) for i in range(9)]

f, ax = plt.subplots(figsize=(15, 5))
num=1 # Pick a number in the range 1-10

ax.plot([drop5.iloc[i]['frame'] for i in rises5[num]], 
        [drop5.iloc[i]['x'] for i in rises5[num]])

ax.plot([drop5.iloc[i]['frame'] for i in falls5[num]], 
        [drop5.iloc[i]['x'] for i in falls5[num]])

plt.show()
vup5 = [linefit(drop5, rises5[i])[0] for i in range(len(rises5))]
vuperr5 = [linefit(drop5, rises5[i])[1] for i in range(len(rises5))]
vdown5 = [linefit(drop5, falls5[i])[0] for i in range(len(falls5))]
vdownerr5 = [linefit(drop5, falls5[i])[1] for i in range(len(falls5))]
f, ax = plt.subplots(figsize = (15, 10))
ax.bar(np.arange(len(vup5)), vup5, yerr = vuperr5)
ax.bar(np.arange(len(vdown5)), vdown5, yerr = vdownerr5)
ax.set(ylabel='Velocity (Pixels/Frame)', xticks=np.arange(len(vup5)))
ax.legend(['Rises', 'Falls'])
plt.show()

vf1 = abs(np.array(vdown1) * 1.9966*10**(-4)) #Using measured pixel-mm conversion rate
vr1 = abs(np.array(vup1) * 1.9966*10**(-4))
f, ax = plt.subplots(figsize = (15, 5))
ax.bar(np.arange(len(vf1)), vf1, alpha = 0.5)
ax.bar(np.arange(len(vr1)), vr1, alpha = 0.5, zorder=0)
ax.set(ylabel='Velocity (m/s)', xticks=np.arange(len(vr1)))
ax.legend(['$v_f$', '$v_r$'])
plt.show()
def charge1(vf, vr): 
    b = 8.2*10**(-3)
    p = 101591.4 
    eta = 1.8512 * 10 **(-5) 
    row = 886
    g = 9.81
    d = 0.0075
    V = 500
    
    q = (4*np.pi/3)*((np.sqrt((b/(2*p))**2 + 9*eta*vf/(2*row*g))-b/(2*p))**3)*(row*g*d*(vf+vr))/(V*vf)
    return q
charges1 = np.array([charge1(np.average(vf1), v) for v in vr1])
print(np.average(charges1[charges1 < 4e-19])) # What's the average of my lowest set of values?
f, ax = plt.subplots()
ax.bar(np.arange(len(charges1)), charges1)
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(15)]) for charge in charges1]))

xrange = np.arange(1e-19, 4e-19, 0.001e-19)
vals = [res(x) for x in xrange]
elem = min(xrange[np.where(vals==min(vals))])
print(elem)
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
f, ax = plt.subplots(figsize=(15, 5))
ax.hist(charges1/(elem), bins=50, align='mid')
ax.set(xticks=range(10))
plt.show()
vf2 = abs(np.array(vdown2) * 1.9966*10**(-4))
vr2 = abs(np.array(vup2) * 1.9966*10**(-4))
f, ax = plt.subplots(figsize = (15, 5))
ax.bar(np.arange(len(vf2)), vf2, alpha = 0.5)
ax.bar(np.arange(len(vr2)), vr2, alpha = 0.5, zorder=0)
ax.set(ylabel='Velocity (m/s)', xticks=np.arange(len(vr2)))
ax.legend(['$v_f$', '$v_r$'])
plt.show()
def charge2(vf, vr): 
    b = 8.2*10**(-3)
    p = 101591.4 
    eta = 1.8512 * 10 **(-5) 
    row = 886
    g = 9.81
    d = 0.0075
    V = 500
    
    q = (4*np.pi/3)*((np.sqrt((b/(2*p))**2 + 9*eta*vf/(2*row*g))-b/(2*p))**3)*(row*g*d*(vf+vr))/(V*vf)
    return q
charges2 = np.array([charge2(np.average(vf2), v) for v in vr2])
print(np.average(charges2[charges2 < 0.25e-18])) # What's the average of my lowest set of values?
f, ax = plt.subplots()
ax.bar(np.arange(len(charges2)), charges2)
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(15)]) for charge in charges2]))

xrange = np.arange(1e-19, 4e-19, 0.001e-19)
vals = [res(x) for x in xrange]
elem=min(xrange[np.where(vals==min(vals))])
print(elem)
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
f, ax = plt.subplots(figsize=(15, 5))
ax.hist(charges2/(elem), bins=20, align='mid')
ax.set(xticks=range(10))
plt.show()
vf3 = abs(np.array(vdown3) * 1.9966*10**(-4))
vr3 = abs(np.array(vup3) * 1.9966*10**(-4))
def charge3(vf, vr): 
    b = 8.2*10**(-3)
    p = 101591
    eta = 1.8463 * 10 **(-5) 
    row = 886
    g = 9.81
    d = 0.0075
    V = 500
    
    q = (4*np.pi/3)*((np.sqrt((b/(2*p))**2 + 9*eta*vf/(2*row*g))-b/(2*p))**3)*(row*g*d*(vf+vr))/(V*vf)
    return q
charges3 = np.array([charge3(np.average(vf3), v) for v in vr3])
print(np.average(charges3[charges3 < 0.25e-18])) # What's the average of my lowest set of values?
f, ax = plt.subplots()
ax.bar(np.arange(len(charges3)), charges3, alpha=0.7)
ax.set(ylim=[0, 1.6e-18])
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(15)]) for charge in charges3]))

xrange = np.arange(1e-19, 3e-19, 0.001e-19)
vals = [res(x) for x in xrange]
elem=min(xrange[np.where(vals==min(vals))])
print(elem)
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
f, ax = plt.subplots(figsize=(15, 5))
ax.hist(charges3/(elem), bins=100, align='mid')
ax.set(xticks=range(10))
plt.show()
vf4 = abs(np.array(vdown4) * 1.9966*10**(-4))
vr4 = abs(np.array(vup4) * 1.9966*10**(-4))
def charge4(vf, vr): 
    b = 8.2*10**(-3)
    p = 101591
    eta = 1.8463 * 10 **(-5) 
    row = 886
    g = 9.81
    d = 0.0075
    V = 500
    
    q = (4*np.pi/3)*((np.sqrt((b/(2*p))**2 + 9*eta*vf/(2*row*g))-b/(2*p))**3)*(row*g*d*(vf+vr))/(V*vf)
    return q
charges4 = np.array([charge4(np.average(vf4), v) for v in vr4])
print(np.average(charges4[charges4 < 4e-18])) # What's the average of my lowest set of values?
f, ax = plt.subplots()
ax.bar(np.arange(len(charges4)), charges4, alpha=0.7)
ax.set(ylim=[0, 1.6e-18])
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(15)]) for charge in charges4]))

xrange = np.arange(1e-19, 3e-19, 0.001e-19)
vals = [res(x) for x in xrange]
elem=min(xrange[np.where(vals==min(vals))])
print(elem)
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
f, ax = plt.subplots(figsize=(15, 5))
ax.hist(charges4/(elem), bins=100, align='mid')
ax.set(xticks=range(10))
plt.show()
vf5 = abs(np.array(vdown5) * 1.9966*10**(-4))
vr5 = abs(np.array(vup5) * 1.9966*10**(-4))
def charge5(vf, vr): 
    b = 8.2*10**(-3)
    p = 101591
    eta = 1.8463 * 10 **(-5) 
    row = 886
    g = 9.81
    d = 0.0075
    V = 500
    
    q = (4*np.pi/3)*((np.sqrt((b/(2*p))**2 + 9*eta*vf/(2*row*g))-b/(2*p))**3)*(row*g*d*(vf+vr))/(V*vf)
    return q
charges5 = np.array([charge5(np.average(vf5), v) for v in vr5])
print(np.average(charges5[charges5 < 2.5e-18])) # What's the average of my lowest set of values?
f, ax = plt.subplots()
ax.bar(np.arange(len(charges5)), charges5, alpha=0.7)
ax.set(ylim=[0, 1.6e-18])
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(15)]) for charge in charges5]))

xrange = np.arange(1e-19, 3e-19, 0.001e-19)
vals = [res(x) for x in xrange]
elem=min(xrange[np.where(vals==min(vals))])
print(elem)
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
f, ax = plt.subplots(figsize=(15, 5))
ax.hist(charges5/(elem), bins=100, align='mid')
ax.set(xticks=range(10))
plt.show()

charges = np.concatenate((charges1, charges2, charges3, charges4, charges5), axis=0)

f, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
ax1.bar(np.arange(len(charges)), charges)
ax1.set(ylim=[0,1.5e-18])
ax2.hist(charges, bins=100)
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(30)]) for charge in charges]))

xrange = np.arange(1.3e-19, 9e-19, 0.001e-19)
vals = [res(x) for x in xrange]
print('Elementary charge: {0}'.format(min(xrange[np.where(vals==min(vals))])))
print('Standard deviation: {0}'.format(min(np.sqrt(np.array(vals)))))
print('Standard deviation as percentage of e: {0}'.format(min(np.sqrt(np.array(vals)))/min(xrange[np.where(vals==min(vals))])))
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
dev = np.sqrt(np.array([min([(charge-n*1.606e-19)**2 for n in np.arange(30)]) for charge in charges]))
plt.bar(range(len(charges)),dev)
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(30)]) for charge in charges[dev < 6e-20]]))

xrange = np.arange(1.3e-19, 9e-19, 0.001e-19)
vals = [res(x) for x in xrange]
print('Elementary charge: {0}'.format(min(xrange[np.where(vals==min(vals))])))
print('Standard deviation: {0}'.format(min(np.sqrt(np.array(vals)))))
print('Standard deviation as percentage of e: {0}'.format(min(np.sqrt(np.array(vals)))/min(xrange[np.where(vals==min(vals))])))
f, ax = plt.subplots()
ax.plot(xrange, [res(x) for x in xrange])
plt.show()
f, ax = plt.subplots(figsize=(7, 4))
num=50
ax.hist(np.delete(charges1, [6])/1.606e-19, bins=num, alpha=0.5, range=(0, 10), zorder=2)
ax.hist(np.delete(charges2, [29-len(charges1),30-len(charges1),31-len(charges1)])/1.606e-19, bins=num, 
        range=(0, 10), alpha=0.5)
ax.hist(charges3/1.606e-19, bins=num, alpha=0.5, range=(0, 10), zorder=0)
ax.hist(charges4/1.606e-19, bins=num, alpha=0.5, range=(0, 10), zorder=0)
ax.hist(charges5/1.606e-19, bins=num, alpha=0.5, range=(0, 10), zorder=0)
ax.hist(charges5/1.606e-19, bins=num, alpha=0.5, range=(0, 10), zorder=0)

ax.set(xticks=np.arange(10), xlim=[0,10], title='Histogram of droplet charge', 
       xlabel=r'Droplet charge / $e$', ylabel='Count number', yticks=range(0,13))
ax.legend(['Trial 1', 'Trial 2','Trial 3','Trial 4','Trial 5'])
ax.text(7, 8, r'$e=1.615 \times 10^{-19}$ C')
plt.show()