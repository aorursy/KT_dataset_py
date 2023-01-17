import pandas as pd # to store data as dataframe
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from scipy.optimize import curve_fit # for the signal and background fit
samples_list = ['data_A'] # add 'data_B','data_C','data_D' later if you want

fraction = 0.8 # increase this later if you want

DataFrames = {} # define empty dictionary to hold dataframes
for s in samples_list: # loop over samples
    DataFrames[s] = pd.read_csv('/kaggle/input/gamgam-csv/'+s+'.csv', index_col='entry') # read .csv file
all_data = pd.concat(DataFrames) # merge DataFrames into one
def calc_myy(photon_pt_1,photon_eta_1,photon_phi_1,photon_E_1,
             photon_pt_2,photon_eta_2,photon_phi_2,photon_E_2):
    # 1st photon is _1, 2nd photon is _2 etc
    
    # sumE = sum of energy
    sumE = photon_E_1 + photon_E_2
    
    px_1 = photon_pt_1*np.cos(photon_phi_1) # x-momentum of photon_1
    # px_2 = x-momentum of photon_2
    px_2 = photon_pt_2*np.cos(photon_phi_2)
    
    py_1 = photon_pt_1*np.sin(photon_phi_1) # y-momentum of photon_1
    # py_2 = y-momentum of photon_2
    py_2 = photon_pt_2*np.sin(photon_phi_2)
    
    pz_1 = photon_pt_1*np.sinh(photon_eta_1) # z-momentum of photon_1
    # pz_2 = z-momentum of photon_2
    pz_2 = photon_pt_2*np.sinh(photon_eta_2)
    
    # sumpx = sum of x-momenta
    sumpx = px_1 + px_2
    
    # sumpy = sum of y-momenta
    sumpy = py_1 + py_2
    
    # sumpz = sum of z-momenta
    sumpz = pz_1 + pz_2
    
    # sump = magnitude of total momentum
    sump = np.sqrt(sumpx**2 + sumpy**2 + sumpz**2)
    
    # myy = invariant mass from M^2 = E^2 - p^2
    myy = np.sqrt(sumE**2 - sump**2)
    
    return myy
# myy is calculated for each row in the data
all_data['myy'] = np.vectorize(calc_myy)(all_data['photon_pt_1'],
                                         all_data['photon_eta_1'],
                                         all_data['photon_phi_1'],
                                         all_data['photon_E_1'],
                                         all_data['photon_pt_2'],
                                         all_data['photon_eta_2'],
                                         all_data['photon_phi_2'],
                                         all_data['photon_E_2'])
# xmin = x-axis minimum in GeV from the Higgs discovery diphoton graph 
xmin = 100
# xmax = x-axis minimum in GeV from the Higgs discovery diphoton graph 
xmax = 160
# step_size = x-axis separation between data points in GeV from the Higgs discovery diphoton graph
step_size = 2

bin_edges = np.arange(start=xmin, # The interval includes this value
                 stop=xmax+step_size, # The interval doesn't include this value
                 step=step_size ) # Spacing between values

bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                        stop=xmax+step_size/2, # The interval doesn't include this value
                        step=step_size ) # Spacing between values
print(all_data['myy'])
print(type(bin_edges))
print(bin_edges)
print(type(bin_centres))
print(bin_centres)
def plot_data():
    
    #####################
    # Filling a histogram
    # For you to complete
    # Complete the lines below
    histogrammed_data,_ = np.histogram( all_data['myy'],
                                       bins=bin_edges )
    # For you to complete
    #####################

    histogrammed_data_errors = np.sqrt( histogrammed_data ) # statistical error on the data

    #####################
    # Plot the data points
    # For you to complete
    # Complete the lines below then uncomment them
    plt.errorbar(x=bin_centres, 
                 y=histogrammed_data, 
                 yerr=histogrammed_data_errors,
                 label='Data',
                 fmt='ko' ) # 'k' means black and 'o' means circles
    # For you to complete
    #####################
    
    return histogrammed_data
plot_data()
# Select eta outside the barrel/end-cap transition region
# you can think of eta as the photon's position in the detector
# paper: "excluding the calorimeter barrel/end-cap transition region 1.37 < |η| < 1.52"
def select_eta(photon_eta_1,photon_eta_2):
# want to keep events where absolute value of photon_eta is outside the range 1.37 to 1.52
    # if absolute value of either photon_eta between 1.37 and 1.52: return False
    if abs(photon_eta_1)>1.37 and abs(photon_eta_1)<1.52: return False
    if abs(photon_eta_2)>1.37 and abs(photon_eta_2)<1.52: return False
    else: return True
    
all_data = all_data[ np.vectorize(select_eta)(all_data.photon_eta_1,all_data.photon_eta_2) ]


# Select photons with high pt
# pt is related to the photon's momentum
# paper: "The leading (sub-leading) photon candidate is required to have ET > 40 GeV (30 GeV)"
def select_pt(photon_pt_1,photon_pt_2):
# want to keep events where photon_pt_1>40 GeV and photon_pt_2>30 GeV
    # if photon_pt_1 greater than 40 GeV and photon_pt_2 greater than 30 GeV: return True
    #if photon_pt_1>40 and photon_pt_2>30: return True
    #if photon_pt...
    if True: return True
    else: return False
    
all_data = all_data[ np.vectorize(select_pt)(all_data.photon_pt_1,all_data.photon_pt_2) ]


# Select photons with low noise around them
# you can think of etcone20 as how much noise is going on around the photon
# paper: "Photon candidates are required to have an isolation transverse energy of less than 4 GeV"
def select_etcone20(photon_etcone20_1,photon_etcone20_2):
# want to keep events where isolation eT<4 GeV
    # if both photon_etcone20 less than 4 GeV: return True
    #if photon_etcone20_1<4 and photon_etcone20_2<4: return True
    #if photon_etcone20_1...
    if True: return True
    else: return False
    
all_data = all_data[ np.vectorize(select_etcone20)(all_data.photon_etcone20_1,all_data.photon_etcone20_2) ]


# Select tightly identified photons
# isTightID==True means a photon more likely to be a real photon, and not some error in the detector
# paper: "Photon candidates are required to pass identification criteria"
def select_isTightID(photon_isTightID_1,photon_isTightID_2):
# isTightID==True means a photon identified as being well reconstructed
# want to keep events where True for both photons
    # if both photon_isTightID are True: return True
    #if photon_isTightID_1==True and photon_isTightID_2==True: return True
    #if photon_isTightID_1...
    if True: return True
    else: return False

all_data = all_data[ np.vectorize(select_isTightID)(all_data.photon_isTightID_1,all_data.photon_isTightID_2) ]
plot_data()
def func(x, c0, c1, c2, c3, c4, A, mu, sigma): # define function for polynomial + Gaussian
    return c0 + c1*x + c2*x**2+ c3*x**3 + c4*x**4 + A*np.exp(-0.5*((x-mu)/sigma)**2)

data = plot_data() # draw a plot
errors = np.sqrt(data) # get the errors on the y values

# data fit
popt,_ = curve_fit(func, # function to fit
                   bin_centres, # x
                   data, # y
                   p0=[data.max(),0,0,0,0,91.7,125,2.4], # initial guesses for the fit parameters
                   sigma=errors) # errors on y

# background part of fit
c0 = popt[0] # c0 of c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
c1 = popt[1] # c1 of c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
c2 = popt[2] # c2 of c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
c3 = popt[3] # c3 of c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
c4 = popt[4] # c4 of c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
# get the background only part of the fit to data
background_fit = c0 + c1*bin_centres + c2*bin_centres**2 + c3*bin_centres**3 + c4*bin_centres**4

A = popt[5] # amplitude of Gaussian
mu = popt[6] # centre of Gaussian
sigma = popt[7] # width of Gaussian
fit = func(bin_centres,c0,c1,c2,c3,c4,A,mu,sigma) # call func with fitted parameters

# plot the signal + background fit
plt.plot(bin_centres, # x
         fit, # y
         '-r', # single red line
         label='Sig+Bkg Fit ($m_H=125$ GeV)' )
# plot the background only fit
plt.plot(bin_centres, # x
         background_fit, # y
         '--r', # dashed red line
         label='Bkg (4th order polynomial)' )

plt.ylabel( 'Events' ) # write y-axis label for main axes
plt.ylim( bottom=0 ) # set the y axis limit for the main axes
plt.xlabel(r'di-photon invariant mass $\mathrm{m_{\gamma\gamma}}$ [GeV]') # x-axis label

# draw the legend
plt.legend()

print('gaussian centre = '+str(mu))
print('gaussian sigma = '+str(sigma))
print(fit)
print(data)
print(errors)
# calculate chi squared
fit_minus_data = fit - data
fit_minus_data_over_errors = fit_minus_data/errors
fit_minus_data_over_errors_squared = fit_minus_data_over_errors**2
chisq = sum(fit_minus_data_over_errors_squared)

print('chi^2 = '+str(chisq))
