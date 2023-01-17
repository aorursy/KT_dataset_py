import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit as fit
from skimage import io
%matplotlib inline
drop1=pd.read_csv('../input/millikan/trial1_track.csv')
drop2=pd.read_csv('../input/millikan/trial2_track.csv')
drop3=pd.read_csv('../input/millikan/trial3_track.csv')

drop1.head()
peak1 = signal.find_peaks_cwt(drop1['x'], np.arange(1,300)) #Finding peaks
val1 = signal.find_peaks_cwt(-drop1['x'], np.arange(1, 500)) #Finding valleys

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
peak2 = np.delete(peak2, [9, 19, peak2.size-1])
val2 = np.delete(val2, [9, 19])
f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(drop2['frame'], drop2['x'])
ax.scatter(peak2, [drop2.iloc[i]['x'] for i in peak2])
ax.scatter(val2, [drop2.iloc[i]['x'] for i in val2])
ax.set(title='Positional data in vertical direction of 20 minute trial', 
       ylabel='Position in vertical direction in pixels', 
       xlabel='Frame')
plt.show()
rises2 = [np.arange(peak2[i], val2[i]) for i in range(len(peak2))]
falls2 = [np.arange(val2[i], peak2[i+1]) for i in range(len(peak2)-1)]

f, ax = plt.subplots(figsize=(15, 5))
num = 20 #Max num is 32

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
f, ax = plt.subplots(figsize=(20, 10))
ax.scatter(drop3['frame'], drop3['x'])
ax.scatter(peak3a, [drop3.iloc[i]['x'] for i in peak3a])
ax.scatter(val3a, [drop3.iloc[i]['x'] for i in val3a])

for i in range(len(peak3a)): 
    ax.annotate('{0}'.format(i), xy=(peak3a[i], drop3.iloc[peak3a[i]]['x']))
for i in range(len(val3a)): 
    ax.annotate('{0}'.format(i), xy=(val3a[i], drop3.iloc[val3a[i]]['x']))
peak3 = np.delete(peak3a, [6, 28, len(peak3a)-1])
val3 = np.delete(val3a, [6, 28])

f, ax = plt.subplots(figsize = (20, 10))
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
num= 10 # Change to view different rises/falls
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
frame = 20
im = io.imread('../input/trial-2-image-set/trial 2 image set/Trial 2 Image Set/{0}{1}.jpg'.format('0'*(4-len(str(frame+678))), frame+678))
pat = im[-int(drop2.iloc[frame]['y'])-10:-int(drop2.iloc[frame]['y'])+10, int(drop2.iloc[frame]['x'])-10:int(drop2.iloc[frame]['x'])+10]

pat[pat<0.6*pat.max()]=0 # Mask value
plt.imshow(pat)
plt.colorbar()
plt.show()
def track2_std(frame):
    im = io.imread('../input/trial-2-image-set/trial 2 image set/Trial 2 Image Set/{0}{1}.jpg'.format('0'*(4-len(str(frame+678)))*(len(str(frame+678))<5), frame+678))
    pat = im[-int(drop2.iloc[frame]['y'])-10:-int(drop2.iloc[frame]['y'])+10, int(drop2.iloc[frame]['x'])-10:int(drop2.iloc[frame]['x'])+10]
    y, x = np.where(pat > 0.6*pat.max())
    stdx = np.sqrt(np.mean(np.array([(i-10)*(i-10) for i in x])))
    stdy = np.sqrt(np.mean(np.array([(i-10)*(i-10) for i in y])))
    return stdx, stdy
sample = np.arange(0, len(drop2), 30)

stdx = [track2_std(int(s))[0] for s in sample]
stdy = [track2_std(int(s))[1] for s in sample]

print('Standard deviation of droplet pixels from tracking point: \n Vertical direction: {0} \n Horizontal direction: {1}'.format(np.average(stdx), np.average(stdy)))
f, ax = plt.subplots()
ax.hist(stdy, alpha = 0.5, bins=10)
ax.hist(stdx, alpha = 0.5, bins=10)
ax.set(title='Standard deviation of droplet pixels from tracking point', 
       xlabel='Standard deviation of droplet pixels from tracker location (pixels)', ylabel='Count number')
ax.legend(['Horizontal direction', 'Vertical direction'])
plt.show()
f, ax = plt.subplots(figsize=(20, 10))

ax.scatter(sample, stdx, alpha =0.5)

num=int(drop2.frame.max()/300) # Number of bars
bins = np.linspace(0, sample.max(), num)
vals = [np.average([stdx[i] for i in np.where((sample >bins[i]) & (sample < bins[i+1]))[0]]) for i in range(len(bins)-1)]
ax.bar(left=bins[:-1], height=vals, width=(2/3)*sample.max()/num, zorder=0, align='edge')

ax.set(xlim=[0, drop2.frame.max()], ylabel='Standard deviation of droplet from tracker position', 
       xlabel='Frame number', title='Standard deviation of droplet pixels from tracking point versus frame number')
plt.show()
sigma2_rise = np.array([[stdx[int(np.where((i >= sample) & (i < sample+30))[0])] for i in rises2[j]] for j in range(len(rises2))])
sigma2_fall = np.array([[stdx[int(np.where((i >= sample) & (i < sample+30))[0])] for i in falls2[j]] for j in range(len(falls2))])
vup2 = [linefit(drop2, rises2[i])[0] for i in range(len(rises2))]
vuperr2 = [linefit(drop2, rises2[i])[1] for i in range(len(rises2))]
vdown2 = [linefit(drop2, falls2[i])[0] for i in range(len(falls2))]
vdownerr2 = [linefit(drop2, falls2[i])[1] for i in range(len(falls2))]
frame = 10000
im = io.imread('../input/trial-3-images/trial 3 image set/Trial 3 Image Set/{0}{1}.jpg'.format('0'*(4-len(str(frame+21))), frame+21))
pat = im[-int(drop3.iloc[frame]['y'])-10:-int(drop3.iloc[frame]['y'])+10, int(drop3.iloc[frame]['x'])-10:int(drop3.iloc[frame]['x'])+10]

pat[pat<0.5*pat.max()]=0 # Mask value
plt.imshow(pat)
plt.colorbar()
plt.show()
def track3_std(frame):
    im = io.imread('../input/trial-3-images/trial 3 image set/Trial 3 Image Set/{0}{1}.jpg'.format('0'*(4-len(str(frame+21)))*(len(str(frame+21))<5), frame+21))
    pat = im[-int(drop3.iloc[frame]['y'])-10:-int(drop3.iloc[frame]['y'])+10, int(drop3.iloc[frame]['x'])-10:int(drop3.iloc[frame]['x'])+10]
    y, x = np.where(pat > 0.6*pat.max())
    stdx = np.sqrt(np.mean(np.array([(i-10)*(i-10) for i in x])))
    stdy = np.sqrt(np.mean(np.array([(i-10)*(i-10) for i in y])))
    return stdx, stdy
sample = np.arange(0, len(drop3), 30)

stdx = [track3_std(int(s))[0] for s in sample]
stdy = [track3_std(int(s))[1] for s in sample]

print('Standard deviation of droplet pixels from tracking point: \n Vertical direction: {0} \n Horizontal direction: {1}'.format(np.average(stdx), np.average(stdy)))
f, ax = plt.subplots()
ax.hist(stdy, alpha = 0.5)
ax.hist(stdx, alpha = 0.5)
ax.set(title='Standard deviation of droplet pixels from tracking point', 
       xlabel='Standard deviation of droplet pixels from tracker location (pixels)', ylabel='Count number')
ax.legend(['Horizontal direction', 'Vertical direction'])
plt.show()
f, ax = plt.subplots(figsize=(20, 10))
ax.scatter(sample, stdx, alpha =0.5)

num=int(drop3.frame.max()/300) 
bins = np.linspace(0, sample.max(), num)
vals = [np.average([stdx[i] for i in np.where((sample >bins[i]) & (sample < bins[i+1]))[0]]) for i in range(len(bins)-1)]
ax.bar(left=bins[:-1], height=vals, width=(2/3)*sample.max()/num, zorder=0, align='edge')
ax.set(xlim=[0, drop3.frame.max()], ylabel='Standard deviation of droplet from tracker position', xlabel='Frame number')
plt.show()
print('Frame with highest vertical standard deviation: #{0}'.format(int(sample[np.where(stdx == max(stdx))])))

frame = int(sample[np.where(stdx == max(stdx))])
im = io.imread('../input/trial-3-images/trial 3 image set/Trial 3 Image Set/{0}{1}.jpg'.format('0'*(4-len(str(frame+21))), frame+21))
pat = im[-int(drop3.iloc[frame]['y'])-10:-int(drop3.iloc[frame]['y'])+10, int(drop3.iloc[frame]['x'])-10:int(drop3.iloc[frame]['x'])+10]
pat[pat<0.5*pat.max()]=0 # Mask value

f, ax = plt.subplots()
plt.imshow(pat)
plt.colorbar()
plt.show()
print('Frames with lowest vertical standard deviation:\n{0}.'.format(', '.join([str(i) for i in sample[np.where(stdx == min(stdx))]])))

zeros = sample[np.where(stdx == min(stdx))]

frame = zeros[11] # Pick any number between 0 and 11

print('\nDisplaying frame #%i:' % frame)

im = io.imread('../input/trial-3-images/trial 3 image set/Trial 3 Image Set/{0}{1}.jpg'.format('0'*(4-len(str(frame+21))), frame+21))
pat = im[-int(drop3.iloc[frame]['y'])-10:-int(drop3.iloc[frame]['y'])+10, int(drop3.iloc[frame]['x'])-10:int(drop3.iloc[frame]['x'])+10]
pat[pat<0.5*pat.max()]=0 # Mask value

plt.imshow(pat)
plt.colorbar()
plt.show()
yvel = drop2.y.shift(-1)-drop2.y # Y velocity in one frame (dx)
# Printing average magnitude of yvel (excluding final 100 frames)
print('Average y velocity magnitude (dy): {0}'.format(np.average(np.sqrt(yvel[:-100]*yvel[:-100])))) 
f, ax = plt.subplots(figsize=(15, 10))
yacc = yvel.shift(-1)-yvel # Y acceleration over one frame (dx^2)
#Printing average magnitude of yacc (excluding final 100 frames)
print('Average y acceleration magnitude (dy^2): {0}'.format(np.average(np.sqrt(yacc[:-100]*yacc[:-100]))))
ax.plot(yvel, alpha = 0.5)
ax.plot(yacc, alpha=0.5)
ax.legend(['Velocity', 'Acceleration'])
ax.set(ylim = [-5, 5])
plt.show()
sample = np.arange(0, len(drop2), 30) # Sampling once per second

segments_down = [np.where((sample > peak2[i]) & (sample < val2[i])) for i in range(len(peak2))]
segments_up = [np.where((sample < peak2[i+1]) & (sample > val2[i])) for i in range(len(peak2)-1)]

vels_up=[np.average([yvel.iloc[i] for i in segments_down[arr]]) for arr in range(len(peak2))]
vels_down = [np.average([yvel.iloc[i] for i in segments_up[arr]]) for arr in range(len(peak2)-1)]

accs_up=[np.average([yacc.iloc[i] for i in segments_down[arr]]) for arr in range(len(peak2))]
accs_down = [np.average([yacc.iloc[i] for i in segments_up[arr]]) for arr in range(len(peak2)-1)]

f, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize = (15, 15))
ax1.bar(np.arange(len(vels_down)), vels_down, alpha = 0.7)
ax1.bar(np.arange(len(vels_up)), vels_up, alpha = 0.7)
ax1.set(ylabel='Average horizontal velocity during rises and falls', 
       xticks=range(len(vels_up)))
ax1.legend(['Rises', 'Falls'], loc='lower right')

ax2.bar(np.arange(len(accs_down)), accs_down, alpha = 0.7)
ax2.bar(np.arange(len(accs_up)), accs_up, alpha = 0.7)
ax2.set(ylabel='Average horizontal acceleration during rises and falls', 
        xticks= range(len(vels_up)))

plt.show()
print(peak2[20], val2[20])
yvel = drop3.y.shift(-1)-drop3.y # Y velocity in one frame (dx)
# Printing average magnitude of yvel (excluding final 100 frames)
print('Average horizontal velocity: %f' % np.average(np.sqrt(yvel[:-100]*yvel[:-100]))) 
f, ax = plt.subplots(figsize=(15, 10))
yacc = yvel.shift(-1)-yvel # Y acceleration over one frame (dx^2)
#Printing average magnitude of yacc (excluding final 100 frames)
print('Average horizontal acceleration: %f' % np.average(np.sqrt(yacc[:-100]*yacc[:-100])))
ax.plot(yvel, alpha = 0.5)
ax.plot(yacc, alpha=0.5)
ax.set(ylim = [-5, 5])
plt.show()
sample = np.arange(0, len(drop3), 30)

segments_down = [np.where((sample > peak3[i]) & (sample < val3[i])) for i in range(len(peak3))]
segments_up = [np.where((sample < peak3[i+1]) & (sample > val3[i])) for i in range(len(peak3)-1)]

vels_up=[np.average([yvel.iloc[i] for i in segments_down[arr]]) for arr in range(len(peak3))]
vels_down = [np.average([yvel.iloc[i] for i in segments_up[arr]]) for arr in range(len(peak3)-1)]

accs_up=[np.average([yacc.iloc[i] for i in segments_down[arr]]) for arr in range(len(peak3))]
accs_down = [np.average([yacc.iloc[i] for i in segments_up[arr]]) for arr in range(len(peak3)-1)]

f, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize = (15, 15))
ax1.bar(np.arange(len(vels_down)), vels_down, alpha = 0.7)
ax1.bar(np.arange(len(vels_up)), vels_up, alpha = 0.7)
ax1.set(ylabel='Average horizontal velocity during rises and falls', 
       xticks=range(len(vels_up)))
ax1.legend(['Rises', 'Falls'], loc='lower right')

ax2.bar(np.arange(len(accs_down)), accs_down, alpha = 0.7)
ax2.bar(np.arange(len(accs_up)), accs_up, alpha = 0.7)
ax2.set(ylabel='Average horizontal acceleration during rises and falls', 
        xticks= range(len(vels_up)))

plt.show()
vf1 = abs(np.array(vdown1) * 6.104*30*10**(-6)) #Using measured pixel-mm conversion rate
vr1 = abs(np.array(vup1) * 6.104*30*10**(-6))
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
vf2 = abs(np.array(vdown2) * 5.32*30*10**(-6))
vr2 = abs(np.array(vup2) * 5.32*30*10**(-6))
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
print(np.average(charges2[charges2 < 2e-19])) # What's the average of my lowest set of values?
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
ax.hist(charges2/(elem), bins=100, align='mid')
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
charges = np.concatenate((charges1, charges2, charges3), axis=0)

f, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
ax1.bar(np.arange(len(charges)), charges)
ax1.set(ylim=[0,1.5e-18])
ax2.hist(charges, bins=100)
plt.show()
res = lambda elem: np.average(np.array([min([(charge-n*elem)**2 for n in np.arange(30)]) for charge in charges]))

xrange = np.arange(1.3e-19, 4e-19, 0.001e-19)
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

xrange = np.arange(1.3e-19, 4e-19, 0.001e-19)
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

ax.set(xticks=np.arange(10), xlim=[0,10], title='Histogram of droplet charge', 
       xlabel=r'Droplet charge / $e$', ylabel='Count number', yticks=range(0,13))
ax.legend(['Trial 1 (5 minutes)', 'Trial 2 (20 minutes)','Trial 3 (30 minutes)'])
ax.text(7, 8, r'$e=1.606 \times 10^{-19}$ C')
plt.show()