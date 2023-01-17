import numpy as np

import holoviews as hv



hv.extension('matplotlib')
import warnings

warnings.filterwarnings('ignore')
c = 299792458 # m/c

mc = 0.511e6 # eV/c

Qe = 1.60217662e-19 # elementary charge in Coulombs



p0 = 400e6 # eV/c
Ne = 2e10  # Number of electrons/positrons in the beam

N  = 200000 # number of macro-particles in this simulation
print("Bunch charge = %.1f nC" % (Ne*Qe/1e-9))
sigma_z = 0.6 # m

#sigma_z = 1.0e-2 # m -- to test wakefield calculation



sigma_dp = 0.004 # relative momentum spread
z0  = np.random.normal(scale=sigma_z, size=N)

#z0  = np.random.uniform(low=-sigma_z*2, high=sigma_z*2, size=N)

dp0 = np.random.normal(scale=sigma_dp, size=N)
%opts Scatter (alpha=0.01 s=1) [aspect=3 show_grid=True]



dim_z  = hv.Dimension('z',  unit='m', range=(-12,+12))

dim_dp = hv.Dimension('dp', label='100%*$\Delta p/p$', range=(-1.5,+1.5))



%output backend='matplotlib' fig='png' size=200 dpi=100
hv.Scatter((z0,dp0*100), kdims=[dim_z,dim_dp])
def get_I(z, z_bin = 0.05, z_min=-15, z_max=+15):

    # z, z_bin, z_min, z_max in meters

    

    hist, bins = np.histogram( z, range=(z_min, z_max), bins=int((z_max-z_min)/z_bin) )

    Qm = Qe*Ne/N # macroparticle charge in C

    I = hist*Qm/(z_bin/c) # A



    z_centers = (bins[:-1] + bins[1:]) / 2

    

    return z_centers, I
%opts Area [show_grid=True aspect=3] (alpha=0.5)



dim_I = hv.Dimension('I',  unit='A',  range=(0.0,+1.0))



hv.Area(get_I(z0), kdims=[dim_z], vdims=[dim_I])
L = 27.0 # m -- storage ring perimeter

gamma_t = 6.0 # gamma transition in the ring

eta = 1/(gamma_t*gamma_t) - 1/((p0/mc)*(p0/mc))
#N_turns = 1000020

N_turns = 2000

N_plots = 11



h = 1

eVrf = 5e3 # eV

#eVrf = 0.0 # eV

phi0 = np.pi/2



t_plots = np.arange(0,N_turns+1,int(N_turns/(N_plots-1)))



data2plot = {}



z = z0; dp = dp0

for turn in range(0,N_turns+1):

    if turn in t_plots:

        print( "\rturn = %g (%g %%)" % (turn, (100*turn/N_turns)), end="")

        data2plot[turn] = (z,dp)

    

    phi = phi0 - 2*np.pi*h*(z/L)  # phase in the resonator

    

    # 1-turn transformation:

    dp  = dp + eVrf*np.cos(phi)/p0

    z = z - L*eta*dp
def plot_z_dp(turn):

    z, dp = data2plot[turn]

    z_dp = hv.Scatter((z, dp*100), [dim_z,dim_dp])

    z_I  = hv.Area(get_I(z), kdims=[dim_z], vdims=[dim_I])

    return (z_dp+z_I).cols(1)
#plot_z_dp(1000)
items = [(turn, plot_z_dp(turn)) for turn in t_plots]



m = hv.HoloMap(items, kdims = ['Turn'])

m.collate()
def Wake(xi):

    # of course some other wake can be defined here.

    

    fr = 0.3e9 # Hz

    Rs = 1.0e5 # Ohm

    Q  = 5  # quality factor

   

    wr = 2*np.pi*fr

    alpha = wr/(2*Q)

    wr1 = wr*np.sqrt(1 - 1/(4*Q*Q))

    

    W = 2*alpha*Rs*np.exp(alpha*xi/c)*(np.cos(wr1*xi/c) + (alpha/wr1)*np.sin(wr1*xi/c))

    W[xi==0] = alpha*Rs

    W[xi>0] = 0

    

    return W
%opts Curve [show_grid=True aspect=3]



dim_xi   = hv.Dimension('xi', label=r"$\xi$", unit='m')

dim_Wake = hv.Dimension('W',  label=r"$W$", unit='V/pC')



L_wake = 10 # m

dz = 0.04 # m

xi = np.linspace(-L_wake, 0, int(L_wake/dz)) # m

W = Wake(xi)



hv.Curve((xi, W/1.0e12), kdims=[dim_xi], vdims=[dim_Wake])
zc, I = get_I(z0, z_bin=dz)



V = -np.convolve(W, I)*dz/c # V
zV = np.linspace(max(zc)-dz*len(V), max(zc), len(V))
dim_V = hv.Dimension('V', unit='kV', range=(-10,+10))



(hv.Curve((zV, V/1e3), kdims=[dim_z], vdims=[dim_V]) + \

 hv.Area((zc,I), kdims=[dim_z], vdims=[dim_I])).cols(1)
data2plot = {}



#eVrf = 0    # V

#eVrf = 3e3 # V



z = z0; dp = dp0

for turn in range(0,N_turns+1):

    if turn in t_plots:

        print( "\rturn = %g (%g %%)" % (turn, (100*turn/N_turns)), end="")

        data2plot[turn] = (z,dp)

    

    phi = phi0 - 2*np.pi*h*(z/L)  # phase in the resonator

    

    # RF-cavity

    dp  = dp + eVrf*np.cos(phi)/p0

    

    # wakefield:

    zc, I = get_I(z, z_bin=dz) # A

    V = -np.convolve(W, I)*dz/c # V    

    V_s = np.interp(z,zV,V)

    dp  = dp + V_s/p0



    # z after one turn:

    z = z - L*eta*dp
def plot_z_dp(turn):

    z, dp = data2plot[turn]

    z_dp = hv.Scatter((z, dp*100), [dim_z,dim_dp])

    zc, I = get_I(z, z_bin=dz)

    z_I  = hv.Area((zc,I), kdims=[dim_z], vdims=[dim_I])

    V = -np.convolve(W, I)*dz/c # V

    z_V  = hv.Curve((zV, V/1e3), kdims=[dim_z], vdims=[dim_V])

    return (z_dp+z_I+z_V).cols(1)
items = [(turn, plot_z_dp(turn)) for turn in t_plots]



m = hv.HoloMap(items, kdims = ['Turn'])

m.collate()
#np.save("plots.npy", data2plot)