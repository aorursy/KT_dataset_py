import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.compat.python import lzip

from statsmodels.graphics.gofplots import ProbPlot
DIR_DATA = '../input/'

PATH_MPG = DIR_DATA + 'auto-mpg.csv'
mpg_df = pd.read_csv(PATH_MPG)

print(mpg_df.shape)

mpg_df.head(2)
mpg_df.rename(columns={'model year': 'model_year', 'car name': 'car_name'}, inplace=True)

mpg_df.columns
mpg_df.dtypes
mpg_df.loc[mpg_df.origin == 1, 'origin'] = 'usa'

mpg_df.loc[mpg_df.origin == 2, 'origin'] = 'japan'

mpg_df.loc[mpg_df.origin == 3, 'origin'] = 'germany'
for categorical_column_name in ['model_year', 'origin', 'car_name']:

    mpg_df[categorical_column_name] = mpg_df[categorical_column_name].astype('category')



mpg_df.dtypes
for hp_value in mpg_df.horsepower.unique():

    try:

        int(hp_value)

    except ValueError:

        print('Invalid value in horsepower: `{}`'.format(hp_value))
print('Number of missing values in horsepower:', 

      (mpg_df.horsepower == '?').sum())
mpg_df.drop_duplicates(['origin'])
mpg_df = mpg_df[mpg_df.horsepower != '?']

mpg_df['horsepower'] = mpg_df.horsepower.astype(float)

mpg_df.dtypes
sns.pairplot(mpg_df)
mpg_model = smf.ols('mpg ~ cylinders + horsepower + acceleration + origin', mpg_df).fit()

mpg_model.summary()



#TODO: Can also include other variables into the model and further solve multicollinearity.
sns.regplot('mpg', 'horsepower', data=mpg_df, scatter_kws=dict(alpha=.2))
def plot_resid(model):

    """

    Plot a set of residual plots. 

    """    

    y_fitted = model.fittedvalues    # y-hat

    resid = model.resid

    resid_norm = model.get_influence().resid_studentized_internal



    scatter_fmt = dict(    

        c="b",

        ls="None", 

        marker="o",

        ms=4,

        mfc="none",

        mew=.3,

        alpha=.6,

    )



    ref_line_fmt = dict(

        c="g", 

        ls="--", 

        alpha=.8,

    )



    alpha_lowess = .8

    n_tops = 3

    fontsize_outlier = 10





    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 7.5))

    for ax in [ax1, ax2, ax3, ax4]:

        for side in ["top", "right", "left", "bottom"]:

            ax.spines[side].set_visible(False)





    # Residual vs. fitted plot



    def plot_lowess(x, y, ax):

        """

        Plot lowess line and scatters.

        """

        lowess_result = sm.nonparametric.lowess(endog=y, exog=x)

        # Note `lowess`' unusual ordering of positional arguments

        y_fitted_sorted, y_lowess_sorted = (lowess_result[:,0], 

                                            lowess_result[:,1])

        ax.plot(y_fitted_sorted, y_lowess_sorted, alpha=alpha_lowess)

        ax.plot(x, y, **scatter_fmt)



    plot_lowess(y_fitted, resid, ax1)

    ax1.axhline(y=0, **ref_line_fmt)



    ax1.set_xlabel("$Fitted\ values$")

    ax1.set_ylabel("$Residuals$")

    ax1.set_title("Residuals vs. fitted", y=1.02)



    resid_abs_tops = np.abs(resid).sort_values(ascending=False)[:n_tops]

    for index in resid_abs_tops.index:

        ax1.annotate(index, 

                     xy=(y_fitted[index], resid[index]), 

                     fontsize=fontsize_outlier)





    # Q-Q plot



    QQ = ProbPlot(resid_norm)

    QQ.qqplot(lw=10, ax=ax2, **scatter_fmt)

    ax2.set_title('Normal Q-Q', y=1.02)

    ax2.set_xlabel('$Theoretical\ quantiles$')

    ax2.set_ylabel('$Standardized\ residuals$');



    end_pts = lzip(ax2.get_xlim(), ax2.get_ylim())

    end_pts[0] = min(end_pts[0])

    end_pts[1] = max(end_pts[1])

    ax2.plot(end_pts, end_pts, **ref_line_fmt)    # Diagonal line

    ax2.set_xlim(end_pts)

    ax2.set_ylim(end_pts)



    resid_norm_abs = np.abs(resid_norm)

    resid_norm_abs_tops = np.flip(np.argsort(resid_norm_abs), 0)[:n_tops]

    for r, index in enumerate(resid_norm_abs_tops):

        ax2.annotate(index, 

                     xy=(np.flip(QQ.theoretical_quantiles, 0)[r],

                         resid_norm[index]), 

                     fontsize=fontsize_outlier)





    # Scale-location plot



    resid_norm_abs_sqrt = np.sqrt(resid_norm_abs)

    plot_lowess(y_fitted, resid_norm_abs_sqrt, ax3)



    ax3.set_title('Scale-Location', y=1.02)

    ax3.set_xlabel('$Fitted\ values$')

    ax3.set_ylabel('$\sqrt{|Standardized\ Residuals|}$');



    resid_norm_abs_sqrt_tops = np.flip(np.argsort(resid_norm_abs_sqrt), 

                                      0)[:n_tops]

    for index in resid_norm_abs_sqrt_tops:

        ax3.annotate(index, 

                     xy=(y_fitted[index], resid_norm_abs_sqrt[index]),

                     fontsize=fontsize_outlier)





    # Leverage polot



    leverage = model.get_influence().hat_matrix_diag

    cooks = model.get_influence().cooks_distance[0]

    plot_lowess(leverage, resid_norm, ax4)



    ax4.set_title('Residuals vs. leverage', y=1.02)

    ax4.set_xlabel('$Leverage$')

    ax4.set_ylabel('$Standardized\ residuals$')



    leverage_tops = np.flip(np.argsort(cooks), 0)[:n_tops]

    for i in leverage_tops:

        ax4.annotate(i, 

                     xy=(leverage[i], resid_norm[i]), 

                     fontsize=fontsize_outlier)



    xlim = ax4.get_xlim()

    ylim = ax4.get_ylim()

    ax4.set_ylim(*ylim)



    def plot_cook(D, ax):

        """

        Plot Cook's distance countours.

        """        

        x = np.linspace(.001, xlim[1], 50)

        n_paras = len(model.params)

        y = np.sqrt((D*n_paras*(1-x)) / x)

        plt.plot(x, y, **ref_line_fmt)



    plot_cook(.5, ax4)

    plot_cook(1, ax4)



    plt.tight_layout()
plot_resid(mpg_model)