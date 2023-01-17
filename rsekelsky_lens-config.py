import numpy as np

# Incoming beam waist (micron)
w1 = 2 / 2 * 1e3 # Most likely value
wlair = 794.76569 * 10**(-3) # wavelength (micron)

# Focal lengths (micron) 
lenses = np.array([30, 50, 60, 75, 100, 150, 200]) * 1e3  # Thor air-spaced achromatic doublets.
lenses_2 = np.array([7.5, 10, 12.7, 15, 16, 19, 20, 25, 35, 40, 45, 80.3]) * 1e3  # Thor achromatic doublets.
lenses_3 = np.array([25.4]) * 1e3  # Thor singlets.
lenses_4 = np.array([6.4, 38.1, 50.2, 75.6, 88.3]) * 1e3  # Newport singlets.
lenses = np.append(lenses, lenses_2)
lenses = np.append(lenses, lenses_3)
lenses = np.append(lenses, lenses_4)
print(lenses)

# REQUIRED: 88.3

# Desired beam waist
w0e1_des = 45.762 # (micron)
w0e2_des = 51.394 # (micron)

configs = np.zeros((1, 11))

# Telescope
for f1 in lenses:
    for f2 in lenses:
        for f3 in lenses:
            
            w2 = f2/f1 * w1
            w0e1 = wlair * f3 / (np.pi * w2)
            
            for f4 in lenses:
                for f5 in lenses:

                    w0e2 = wlair*f5/(np.pi*w2)

                    dev1 = w0e1 - w0e1_des
                    dev2 = w0e2 - w0e2_des
                    devtot = np.abs(dev1) + np.abs(dev2)

                    if w2==500:
                        if devtot < 3 and np.abs(dev1) < 2 and np.abs(dev2) <2:
                            config = np.array([f1,f2,f3,f4,f5]) * 1e-3 # Convert back to mm
                            result = np.array([w0e1,w0e2,dev1,dev2,devtot, (f1 + f2 + 2*f3 + 2*f4 + 2*f5) / 1e3])
                            configs = np.append(configs, [np.concatenate([config,result])], axis=0)
configs = configs[configs[:,-1].argsort()]
configs = configs[1:]

print("f1","f2","f3","f4","f5","w0e1","w0e2","err1","err2","sum","total_dist")

for row in configs:
    print(['{0:.1f}'.format(row[x]) for x in range(len(row))])
# PAF2-7B beam diamter (units in mm)

import numpy as np

f = 12

# Our laser
lambd = 794.76569 * 10**(-6)

# P1-780A-FC-2 (Marcuse's equation)
MFD = 5.34 * 10**(-3)

# THOR Spec
# lambd = 850 * 10**(-6)
# MFD = 5.0 * 10**(-3)

d = 4 * lambd * f / np.pi / MFD

print("Output beam diamter : %.3f mm" % d)
d = 1.397 # Razor blade
d = 1.354 # Razor blade

MFD = 4*lambd*7.5/(np.pi*d) * 10**3

print("MFD : %.3f micro meter" % MFD)
# PAF2-5B beam diameter

f = 12 # mm

lambd = 794.76569 * 10**(-6)

MFD = 5.607 * 10**(-3)
MFD = 5.434 * 10**(-3)
MFD = 5.34 * 10**(-3)

d = 4 * lambd * f / np.pi / MFD

print("Output beam diamter : %.3f mm" % d)
lambd = 794.76569 * 10**(-6)

r = 40.68 # mm
ds = [7.5, 4.3] # mm
n = 1.5109

for d in ds:
    w0 = ( lambd**2 * d * (r - d) / (np.pi**2 * n**2)  )**(1/4) * 10**3 # microm
    print("Desired beam waist : %.3f micro meter" % (w0))
# Radius of curvature
# Effective Focal Length : 40, 50, 60, 75, 100 ,125, 150
R = np.array([20.6, 25.8, 30.9, 38.6, 51.5, 64.4, 77.3, 45.633])

n = 1.51452 # @ 650 nm
n = 1.50984 # @ 850 nm
n = 1.5108 # @ 795 nm

f = ( (n-1)/R )**(-1)
f