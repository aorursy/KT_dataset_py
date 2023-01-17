import pandas as pd 

import numpy as np 

import numpy.matlib

import math 

import statsmodels.api as sm

from statsmodels.sandbox.regression.gmm import GMM
def nwSE(r, lag, h0):

    

    T = len(r)

    vv = r.var()

    

    for i in range(1, lag):

        cc = np.cov(np.vstack((r[:-i], r[i:])))

        vv = vv + 2*(1-i/lag)*cc[0,1]

        y = math.sqrt(vv)/math.sqrt(T)

        

    return y 
def factorModel(X, y, lag):

    

    est = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})

    

    return est.params, est.bse, est.rsquared_adj
def sigf(b, se, oneT = False):

    

    n = b.size

    b = b.ravel()

    

    se = se.ravel()

    

    # Calculate t-statistics

    t = np.array(b/se).ravel()

    

    # Set critical values 

    if oneT == True: # One-sided test 

        crit_val = np.array([1.282, 1.645, 2.326])

        

    else: # Two-sided test 

        crit_val = np.array([1.645, 1.96, 2.576])

        

    # Initialize output vector 

    y = []

    

    for i in range(0,n): 

        if abs(t[i]) >= crit_val[2]: # Significant at the 1% level 

            y.append('%.2f***' % b[i])

            

        elif (abs(t[i]) >= crit_val[1]) & (abs(t[i]) < crit_val[2]): # Significant at the 5% level

            y.append('%.2f**' % b[i])

        

        elif (abs(t[i]) >= crit_val[0]) & (abs(t[i]) < crit_val[1]): # Significant at the 10% level,

            y.append('%.2f*' % b[i])

        else: # Not significant

            y.append('%.2f' % b[i])

    

    return y 
df = pd.read_excel('../input/datamaster1/Data_master.xlsx', sheet_name= 'T1_ptfs')
# Average excess returns - Inflation corrected

avret = 12*100*df[['LeadR', 'MidR', 'LagR', 'LL', 'LLStrong']].mean()



# Standard errors (Newey-West correction of 24 lags)

avret_se  = 12*df[['LeadR', 'MidR', 'LagR', 'LL', 'LLStrong']].apply(lambda x: nwSE(x.to_numpy()*100,24,0), axis=0)



# Estimate CAPM (Newey-West correction of 24 lags)

X = sm.add_constant(df[['mktrf']]) # Market returns 

CAPM = df[['LeadR1', 'MidR1', 'LagR1', 'LL', 'LLStrong']].apply(lambda y: factorModel(X,y*100,24), axis=0)



# Annualize alphas and standard errors

CAPM_a = 12 * np.array([CAPM[0][0][0], CAPM[1][0][0], CAPM[2][0][0], CAPM[3][0][0], CAPM[4][0][0]])

CAPM_se = 12 * np.array([CAPM[0][1][0], CAPM[1][1][0], CAPM[2][1][0], CAPM[3][1][0], CAPM[4][1][0]])



# Estimate Fama-French 3-factor model (Newey-West correction of 24 lags)

X = sm.add_constant(df[['mktrf', 'smb', 'hml']]) # 3-factor model returns

FF3 = df[['LeadR1', 'MidR1', 'LagR1', 'LL', 'LLStrong']].apply(lambda y: factorModel(X,y*100,24), axis=0)



# Annualize alphas and standard errors 

FF3_a = 12* np.array([FF3[0][0][0], FF3[1][0][0], FF3[2][0][0], FF3[3][0][0], FF3[4][0][0]])

FF3_se = 12* np.array([FF3[0][1][0], FF3[1][1][0], FF3[2][1][0], FF3[3][1][0], FF3[4][1][0]])
# Average excess returns

avRt = np.array([sigf(avret[0], avret_se[0]), 

                 sigf(avret[1], avret_se[1]),

                 sigf(avret[2], avret_se[2]),

                 sigf(avret[3], avret_se[3]),

                 sigf(avret[4], avret_se[4])]).reshape(5)



# CAPM alphas 

CAPM_alpha = np.array([sigf(CAPM_a[0], CAPM_se[0]),

                       sigf(CAPM_a[1], CAPM_se[1]),

                       sigf(CAPM_a[2], CAPM_se[2]), 

                       sigf(CAPM_a[3], CAPM_se[3]),

                       sigf(CAPM_a[4], CAPM_se[4])]).reshape(5)

    

# Fama-French alphas 

FF3_alpha = np.array([sigf(FF3_a[0], FF3_se[0]),

                      sigf(FF3_a[1], FF3_se[1]),

                      sigf(FF3_a[2], FF3_se[2]),

                      sigf(FF3_a[3], FF3_se[3]),

                      sigf(FF3_a[4], FF3_se[4])]).reshape(5)
# Average excess returns

avret_stdE = ' '.join('(%0.2f)'%F for F in avret_se ).split(' ')



# CAPM alphas

CAPM_stdE = ' '.join('(%0.2f)'%F for F in CAPM_se ).split(' ')

  

# Fama-French alphas 

FF3_stdE = ' '.join('(%0.2f)'%F for F in FF3_se ).split(' ')
pd.DataFrame([avRt, avret_stdE, CAPM_alpha, CAPM_stdE, FF3_alpha, FF3_stdE ],

             index = ['Average return', '', 'CAPM a', '', 'FF3 a', ''],

             columns = ['Lead', 'Mid', 'Lag', 'LL', 'LL Strong'])
df = pd.read_excel('../input/datamaster1/Data_master.xlsx', sheet_name= 'T3_ptfs') 
# Average returns

avret = 12*100*df[['LL38', 'LLStrong38', 'LL49', 'LLStrong49']].mean()



# Standard error (Newey-West correction of 12 lags)

avret_se  = 12*df[['LL38', 'LLStrong38', 'LL49', 'LLStrong49']].apply(lambda x: nwSE(x.to_numpy()*100,12,0), axis=0)



# Estimate CAPM

X = sm.add_constant(df[['mktrf']]) # Market returns

CAPM = df[['LL38', 'LLStrong38', 'LL49', 'LLStrong49']].apply(lambda y: factorModel(X,y*100,12), axis=0) 



# Annualize alphas and standard errors 

CAPM_a = 12 * np.array([CAPM[0][0][0], CAPM[1][0][0], CAPM[2][0][0], CAPM[3][0][0]])

CAPM_se = 12 * np.array([CAPM[0][1][0], CAPM[1][1][0], CAPM[2][1][0], CAPM[3][1][0]])



# Estimate Fama-French three-factor model 

X = sm.add_constant(df[['mktrf', 'smb', 'hml']]) # 3-factor model returns

FF3 = df[['LL38', 'LLStrong38', 'LL49', 'LLStrong49']].apply(lambda y: factorModel(X,y*100,12), axis=0) 



# Annualize alphas and standard errors 

FF3_a = 12* np.array([FF3[0][0][0], FF3[1][0][0], FF3[2][0][0], FF3[3][0][0]])

FF3_se = 12* np.array([FF3[0][1][0], FF3[1][1][0], FF3[2][1][0], FF3[3][1][0]])
# Average excess returns

avRt = np.array([sigf(avret[0], avret_se[0]),

                 sigf(avret[1], avret_se[1]),

                 sigf(avret[2], avret_se[2]), 

                 sigf(avret[3], avret_se[3])]).reshape(4)



# CAPM alphas 

CAPM_alpha = np.array([sigf(CAPM_a[0], CAPM_se[0]), 

                       sigf(CAPM_a[1], CAPM_se[1]),

                       sigf(CAPM_a[2], CAPM_se[2]), 

                       sigf(CAPM_a[3], CAPM_se[3])]).reshape(4)



# Fama-French alphas

FF3_alpha = np.array([sigf(FF3_a[0], FF3_se[0]),

                      sigf(FF3_a[1], FF3_se[1]),

                      sigf(FF3_a[2], FF3_se[2]), 

                      sigf(FF3_a[3], FF3_se[3])]).reshape(4)
# Average excess returns

avret_stdE = ' '.join('(%0.2f)'%F for F in avret_se ).split(' ')



# CAPM alphas 

CAPM_stdE = ' '.join('(%0.2f)'%F for F in CAPM_se ).split(' ')



# Fama-French alphas 

FF3_stdE = ' '.join('(%0.2f)'%F for F in FF3_se ).split(' ')
Panels =  ['Panel A: 38 industries', 'Panel B: 49 industries']

portfolios = ['LL', 'LL Strong']



col = pd.MultiIndex.from_product([Panels, portfolios])



pd.DataFrame([avRt, avret_stdE, CAPM_alpha, CAPM_stdE, FF3_alpha, FF3_stdE ],

             index = ['Average return', '', 'CAPM a', '', 'FF3 a', ''],

             columns = col)
df = pd.read_excel('../input/datamaster1/Data_master.xlsx', sheet_name= 'T7_factors')
# Estimate factor model,

X = sm.add_constant(df[['mktrf', 'smb', 'hml', 'rmw', 'cma']]) # Five-factor model returns 

FF5 = factorModel(X,df.LLStrong30,12) 

FF5_b =  np.array([12*100*FF5[0][0], FF5[0][1], FF5[0][2], FF5[0][3], FF5[0][4], FF5[0][5]])

FF5_se =  np.array([12*100*FF5[1][0], FF5[1][1], FF5[1][2], FF5[1][3], FF5[1][4], FF5[1][5]])

FF5_R2 = FF5[2].round(2)



# Check for significance 

FF5_betas = np.array([sigf(FF5_b[0], FF5_se[0],  True),

                      sigf(FF5_b[1], FF5_se[1]),

                      sigf(FF5_b[2], FF5_se[2]),

                      sigf(FF5_b[3], FF5_se[3]),

                      sigf(FF5_b[4], FF5_se[4]),

                      sigf(FF5_b[5], FF5_se[5])]).reshape(6)



# Format standard errors

FF5_stdE = ' '.join('(%0.2f)'%F for F in FF5_se ).split(' ')
# Estimate factor model 

X = sm.add_constant(df[['q_mkt', 'q_me', 'q_ia', 'q_roe']]) # Q-factor model returns 

HXZ = factorModel(X,df.LLStrong30,12) 

HXZ_b =  np.array([12* 100* HXZ[0][0], HXZ[0][1], HXZ[0][2], HXZ[0][3], HXZ[0][4]])

HXZ_se =  np.array([12* 100* HXZ[1][0], HXZ[1][1], HXZ[1][2], HXZ[1][3], HXZ[1][4]])

HXZ_R2 = HXZ[2].round(2)



# Check for significance 

HXZ_betas = np.array([sigf(HXZ_b[0], HXZ_se[0],  True), 

                      sigf(HXZ_b[1], HXZ_se[1]),

                      sigf(HXZ_b[2], HXZ_se[2]), 

                      sigf(HXZ_b[3], HXZ_se[3]), 

                      sigf(HXZ_b[4], HXZ_se[4])]).reshape(5)



# Format standard errors

HXZ_stdE = ' '.join('(%0.2f)'%F for F in HXZ_se ).split(' ')
# Estimate factor model 

X = sm.add_constant(df[['mom']]) # Q-factor model returns 

F4_1 = factorModel(X,df.LLStrong30,12) 

F4_1_b =  np.array([12* 100* F4_1[0][0], F4_1[0][1]])

F4_1_se =  np.array([12* 100* F4_1[1][0], F4_1[1][1]])

F4_1_R2 = F4_1[2].round(2)



# Check for significance 

F4_1_betas = np.array([sigf(F4_1_b[0], F4_1_se[0],  True), sigf(F4_1_b[1], F4_1_se[1])]).reshape(2)



# Format standard errors

F4_1_stdE = ' '.join('(%0.2f)'%F for F in F4_1_se ).split(' ')
# Estimate factor model 

X = sm.add_constant(df[['mktrf.1','mom']]) # Q-factor model returns 

F4_2 = factorModel(X,df.LLStrong30,12) 

F4_2_b =  np.array([12* 100* F4_2[0][0], F4_2[0][1], F4_2[0][2]])

F4_2_se =  np.array([12* 100* F4_2[1][0], F4_2[1][1], F4_2[1][2]])

F4_2_R2 = F4_2[2].round(2)



# Check for significance 

F4_2_betas = np.array([sigf(F4_2_b[0], F4_2_se[0], True),

                       sigf(F4_2_b[1], F4_2_se[1]),

                          sigf(F4_2_b[2], F4_2_se[2])]).reshape(3)

    

# Format standard errors

F4_2_stdE = ' '.join('(%0.2f)'%F for F in F4_2_se ).split(' ')
# Estimate factor model

X = sm.add_constant(df[['mktrf.1', 'mom', 'smb.1', 'hml.1']]) # Carhart factors 

F4 = factorModel(X,df.LLStrong30,12) 

F4_b =  np.array([12* 100* F4[0][0], F4[0][1], F4[0][2], F4[0][3], F4[0][4]])

F4_se =  np.array([12* 100* F4[1][0], F4[1][1], F4[1][2], F4[1][3], F4[1][4]])

F4_R2 = F4[2].round(2)



# Check for significance 

F4_betas = np.array([sigf(F4_b[0], F4_se[0],  True),

                     sigf(F4_b[1], F4_se[1]),

                     sigf(F4_b[2], F4_se[2]),

                     sigf(F4_b[3], F4_se[3]),

                     sigf(F4_b[4], F4_se[4])]).reshape(5)



# Format standard errors

F4_stdE = ' '.join('(%0.2f)'%F for F in F4_se ).split(' ')
Panels =  ['Panel A: 38 industries', 'Panel B: 49 industries']

portfolios = ['LL', 'LL Strong']

col = pd.MultiIndex.from_product([Panels, portfolios])



row1 = ['a',FF5_betas[0], '', 'a', HXZ_betas[0],'', 'a', F4_1_betas[0], F4_2_betas[0], F4_betas[0]]

row2 = ['', FF5_stdE[0], '', '', HXZ_stdE[0], '', '', F4_1_stdE[0],  F4_2_stdE[0],  F4_stdE[0]]

row3 = ['MKT', FF5_betas[1], '', 'MKT', HXZ_betas[1],'', 'MKT', '', F4_2_betas[1], F4_betas[1]]

row4 = ['', FF5_stdE[1], '', '', HXZ_stdE[1],'', '', '', F4_2_stdE[1], F4_stdE[1]]

row5 = ['SMB', FF5_betas[2], '', 'ME', HXZ_betas[2],'', 'MOM', F4_1_betas[1] , F4_2_betas[2], F4_betas[2]]

row6 = ['', FF5_stdE[2], '', '', HXZ_stdE[2],'', '', F4_1_stdE[1], F4_2_stdE[2], F4_stdE[2]]

row7 = ['HML', FF5_betas[3], '', 'I/A', HXZ_betas[3],'', 'SMB', '' , '', F4_betas[3]]

row8 = ['', FF5_stdE[3], '', '', HXZ_stdE[3],'', '', '', '', F4_stdE[3]]

row9 = ['RMW', FF5_betas[4], '', 'ROE', HXZ_betas[4],'', 'HML', '' , '', F4_betas[4]]

row10 = ['', FF5_stdE[4], '', '', HXZ_stdE[4],'', '', '', '', F4_stdE[4]]

row11 = ['CMA', FF5_betas[5], '', '', '','', '', '', '' , '']

row12 = ['', FF5_stdE[5], '', '', '' ,'', '', '', '', '']

row13 = ['Adjusted R2', FF5_R2, '', '', HXZ_R2, '', '', F4_1_R2,  F4_2_R2,  F4_R2]



pd.DataFrame([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13],

             index = ['','','','','','','','','','','','', ''], columns = ['','','','','','','','','',''])
# Riskfree rate 

df = pd.read_excel('../input/datamaster1/Data_master.xlsx', sheet_name= 'T1_ptfs')

rf = df['rf']



# Factors 

F = df[['mktrf', 'smb', 'hml', 'LL']] # Market, Size, Value, LL 

F['LL'] = F['LL'] * 100 

F = F.to_numpy()

T = len(F)



# 30 industries 

ind30 = pd.read_csv('../input/returndata/30_industry_pfs.csv', sep = ';', header = None)  

ind30 = ind30[(ind30[0] >= 197201) & (ind30[0] <= 201212)].to_numpy() # Select sample 

ind30 = np.hstack((ind30[:,1:31] - np.matlib.repmat(rf.to_numpy(), 30, 1).T, F[:,3].reshape(T,1))) # Add LL factor 

ind30 = ind30[:,~np.any(np.isnan(ind30), axis=0)] # Delete columns with NaN values 



# 38 industries 

ind38 = pd.read_csv('../input/returndata/38_industry_pfs.CSV', sep = ';', header = None) 

ind38 = ind38[(ind38[0] >= 197201) & (ind38[0] <= 201212)].to_numpy()

ind38 = np.hstack((ind38[:,1:39] - np.matlib.repmat(rf.to_numpy(), 38, 1).T, F[:,3].reshape(T,1)))

ind38 = ind38[:,~np.any(np.isnan(ind38), axis=0)]



# 49 industries 

ind49 = pd.read_csv('../input/returndata/49_Industry_pfs.csv', sep = ';', header = None) 

ind49 = ind49[(ind49[0] >= 197201) & (ind49[0] <= 201212)].to_numpy()

ind49 = np.hstack((ind49[:,1:50] - np.matlib.repmat(rf.to_numpy(), 49, 1).T, F[:,3].reshape(T,1)))

ind49 = ind49[:,~np.any(np.isnan(ind49), axis=0)]



# 25 BE/ME and Size 

BM_ME_25 = pd.read_csv('../input/returndata/25_book_size_all.csv', sep = ';', header = None) 

BM_ME_25 = BM_ME_25[(BM_ME_25[0] >= 197201) & (BM_ME_25[0] <= 201212)].to_numpy()

BM_ME_25 = np.hstack((BM_ME_25[:,1:26] - np.matlib.repmat(rf.to_numpy(), 25, 1).T, F[:,3].reshape(T,1)))

BM_ME_25 = BM_ME_25[:,~np.any(np.isnan(BM_ME_25), axis=0)]
class gmm_sdf(GMM):

    def momcond(self, params):

        

        # Initialization

        bMKT, bSMB, bHML, bLL = params

        y = self.endog

        x = self.exog

        inst = self.instrument   

        

        # Time series length 

        T = len(y)

        

        # Number of test assets

        N = np.size(y,1)

        

        # (1-bF)

        e = np.ones((T,)) - np.matmul(x,np.array([bMKT, bSMB, bHML, bLL]))

        

        # Moment conditions 

        g = np.multiply(y, np.matlib.repmat(e, N, 1).T)

       

        return g
def sandwich(moms,gmoms, nobs, nmoms):

    S = np.zeros((nmoms,nmoms))



    for t in range(0,nobs):

        S = S + np.matmul(moms[t,:].reshape((nmoms,1)), moms[t,:].reshape((1,nmoms)))



    S = S/nobs



    gginv = np.linalg.inv(np.matmul(gmoms.T,gmoms))

    gSs = np.matmul(np.matmul(gmoms.T, S), gmoms)



    Vb = np.matmul(np.matmul(gginv, gSs), gginv)/nobs

    

    return Vb
def gmm_est(R,F,b0):

    

    ''' 

    R = TxN matrix with test assets

    F = TxK matrix with factor returns 

    b0 = Kx1 vector with initial guess for paramers

    '''

    

    # Number of test assets 

    N = np.size(R,1)



    # Weighting matrix 

    W = np.eye(N)

    

    # Number of observations in time series  

    T = np.size(R,0)



    # Number of factors 

    K = np.size(F,1)



    # Instruments - Set to identity matrix 

    z = np.ones((T, K))



    # Estimate GMM 

    model = gmm_sdf(R, F, z)

    res = model.fit(b0, maxiter=1, inv_weights = W, 

                     optim_method='nm', wargs=dict(centered=False, maxlag = 12), weights_method ='hac')



    # Store GMM parameters and Newey-West standard errors and covariance matrix of b 

    b = res.params

    se_b = res.bse



    # Covariance matrix of b 

    moms = model.momcond(b)

    gmoms = model.gradient_momcond(b)

    Vb = sandwich(moms, gmoms, T, N)

    

    # There is a mistake in the function for covariance matrix from package (no S in formula Vb) 

    # Vb = res.calc_cov_params(moms, -gradmoms, weights=None, use_weights= True, has_optimal_weights= False,

    #                         weights_method='hac', wargs= dict(centered=False, maxlag = 12))

    

    

    # Compute lambdas estimates and Newey-West standard errors

    

    FF = np.zeros((K,K)) # Covariance matrix F

    

    for t in range(0,T):

        FF = FF + np.matmul(F[t,:].reshape((K,1)), F[t,:].reshape((1,K)))



    FF = FF/T



    # lambda: FF*b

    l = np.matmul(FF,b)



    # Covariance matrix lambda

    Vl = np.matmul(np.matmul(FF,Vb),FF)



    # Se of lambda

    se_l =  np.sqrt(np.diagonal(Vl))

    

    return b,se_b, l, se_l
# 30 industries 

b_ind30, se_b_ind30, l_ind30, se_l_ind30  = gmm_est(ind30,F,np.array([0,0,0,0]))



# Determine significance of coefficients 

coef_ind30 = np.hstack([sigf(l_ind30, se_l_ind30), sigf(b_ind30, se_b_ind30)])



# Format standard errors 

se_ind30 = ' '.join('(%0.2f)'%F for F in np.hstack([se_l_ind30, se_b_ind30])).split(' ')



# 38 industries 

b_ind38, se_b_ind38, l_ind38, se_l_ind38  = gmm_est(ind38,F,np.array([0,0,0,0]))



# Determine significance of coefficients 

coef_ind38 = np.hstack([sigf(l_ind38, se_l_ind38), sigf(b_ind38, se_b_ind38)])



# Format standard errors 

se_ind38 = ' '.join('(%0.2f)'%F for F in np.hstack([se_l_ind38, se_b_ind38])).split(' ')



# 49 industries 

b_ind49, se_b_ind49, l_ind49, se_l_ind49  = gmm_est(ind49,F,np.array([0,0,0,0]))



# Determine significance of coefficients 

coef_ind49 = np.hstack([sigf(l_ind49, se_l_ind49), sigf(b_ind49, se_b_ind49)])



# Format standard errors 

se_ind49 = ' '.join('(%0.2f)'%F for F in np.hstack([se_l_ind49, se_b_ind49])).split(' ')



# 25 BE/ME and Size 

b_BM_ME_25, se_b_BM_ME_25, l_BM_ME_25, se_l_BM_ME_25  = gmm_est(BM_ME_25,F,np.array([0,0,0,0]))



# Determine significance of coefficients 

coef_BM_ME_25 = np.hstack([sigf(l_BM_ME_25, se_l_BM_ME_25), sigf(b_BM_ME_25, se_b_BM_ME_25)])



# Format standard errors 

se_BM_ME_25 = ' '.join('(%0.2f)'%F for F in np.hstack([se_l_BM_ME_25, se_b_BM_ME_25])).split(' ')
pd.DataFrame([coef_ind30, se_ind30, coef_ind38, se_ind38, coef_ind49, se_ind49, coef_BM_ME_25, se_BM_ME_25],

             index = ['30 industries', '', '38 industries', '', '49 industries', '', 'BE/ME and Size (25)', ''],

             columns = ['λmkt', 'λsmb', 'λhml', 'λLL', 'bmkt', 'bsmb', 'bhml', 'bll'])