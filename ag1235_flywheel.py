import math

import matplotlib.pyplot as plt

# acrylic g/cm3

density = 1.18 



M = 50 #g

vol = M / density

print('volume (cm3):',vol)



thickness = 4 #mm

thickness *= 0.1 #convert to cm

r = math.sqrt((vol / thickness) / math.pi)

print('radius (mm):', r * 10)



#convert to kg and meters

M *= 1e-3

r *= 1e-2

density *= 1e3



#acrylic yield strngth MPa

yieldstress = 40
#no load voltage and current

v = 1.54

i = 0.25

p = i * v

w = 9000 / (2 * math.pi)

print('constant voltage:', v)

print('no load power:', p)

print('no load rpm:', w * 2 * math.pi)



R = 1.0

K = (v - i * R) / (i * w)



#initial starting current, w = 0

#initial starting current, 

# assume that motor is an inductor

# emf: v = L . di/dt = L.i.w

# compare with back emf term so L = K

# energy stored in inductor e0 = 1/2.LI^2



i0 = v / R

e0 = 1/2 * K * i0 * i0

print('starting current:', i0)



# V = iR + k.i.w

# Ptot = i^2.R + k.i^2..w  

# Vin . i = i^2.R + k.i^2.w 

# Vin = i.R + k.i.w 

# i = Vin / (R + k.w)

# Pout = k.i^2.w



def I1(w, Vin, R, K):

    return Vin / (R + K * w)





def Pout(i, w, K):

    return K * i * i * w
I = 1 / 2.0 * M * r * r

print(I)
#e = 1/2 . I . w2

#p = 1/2 . I . w2 / t 

#w = sqrt(2.p.t/I) ... (1)

#p = T.w

#T = p / w = 1/2 . I . w2 / ( t.w) = 1/2. I . w / t = 1/2 . I . alph



#from (1)

#def ang_vel(p,t,I):

#    return math.sqrt(2.0 * p * t / I)

def ang_vel(e, I):

    return math.sqrt(2 * e / I)



def torque(w,t,I):

    return 1/2 * I * w / t
X = list(range(1,60))

W = list()

T = list()

P = list()

II = list()

EF = list()

w = 0.0

p - 0.0

E = 0.0

i = i0

for t in X:

    if t == 1:

        p = e0

    else:

        i = I1(w, v, R, K)

        p = Pout(i, w, K)

    P.append(p*1500)

    II.append(i*1000)

    EF.append((p / (p + i*R)) * 10000)

    E += p

    w = ang_vel(E,I)

    W.append(w*60/(2.0*math.pi))

    tq = torque(w,t,I)

    #convert to gf . cm

    tq = tq  * 1000.0 / 9.81 * 100.0 

    T.append(tq)

    
T1 = [ t * 100 for t in T]

plt.figure(figsize=(14,7))

plt.plot(X,W, label='w (rpm)')

plt.plot(X,T1, label='torque *100 (gf.cm)')

plt.plot(X,P, label='power *1500 (w)')

plt.plot(X,II, label='current *1000 (Amp)')

plt.legend()

plt.xlabel('time (s)')

plt.show()

del T1
EF = [ ef / 100.0 for ef in EF]

plt.figure(figsize=(14,7))

plt.plot(W,T, label='torque (gf.cm)')

plt.plot(W,EF, label='efficiency (%)')

plt.xlabel('w (angular velocity) rpm')

plt.ylabel('torque')

plt.legend()

plt.show()