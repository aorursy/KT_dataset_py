# coding: utf-8

import numpy as np

import pandas as pd

import os



class LeaderBoardProbing:



    def __init__(self):

        if os.path.exists('new_test.csv.gz'):

            self.test  = pd.read_csv('new_test.csv.gz')

        else:

            self.test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

            sales=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

            # some routine data cleaning code

            #sales=sales[(sales.item_price<100000) & (sales.item_cnt_day<1001)]

            shop_id_map={0:57, 1:58, 10:11}

            sales['shop_id']=sales['shop_id'].apply(lambda x: shop_id_map.get(x, x))

            self.test['shop_id']=self.test['shop_id'].apply(lambda x: shop_id_map.get(x, x))



            pairs={ (a, b) for a, b in zip(sales.shop_id, sales.item_id) }

            items={ a for a in sales.item_id }

            self.test['date_block_num']=34

            self.test['test_group']=[ 2 if (a,b) in pairs else (1 if b in items else 0) for a,b in zip(self.test.shop_id, self.test.item_id)]

            self.test.sort_values('ID', inplace=True)

            self.test.to_csv('new_test.csv.gz', index=False)



        self.test['item_cnt_month']=0.0

        self.n=len(self.test)

        self.n0=sum(self.test.test_group==0)

        self.n1=sum(self.test.test_group==1)

        self.n2=sum(self.test.test_group==2)



    def probe_mean(self):

        """Generate 4 LeaderBoardProbing files, set target to 0 for all three test groups,

        then set target to 1 for only one test group at a time.

        Manually submit the files to obtain public leaderboard scores.

        Then feed the scores to estimate_mean() to obtain mean values for all groups

        and store those means in group_mean()

        """

        os.makedirs('leak', exist_ok=True)

        self.save(self.test, 'leak/Probe000.csv')



        tmp=self.test.copy()

        tmp.loc[tmp.test_group==2, 'item_cnt_month']=1.0

        self.save(tmp, 'leak/Probe001.csv')



        tmp=self.test.copy()

        tmp.loc[tmp.test_group==1, 'item_cnt_month']=1.0

        self.save(tmp, 'leak/Probe010.csv')



        tmp=self.test.copy()

        tmp.loc[tmp.test_group==0, 'item_cnt_month']=1.0

        self.save(tmp, 'leak/Probe100.csv')



    def estimate_mean(self, rmse000, rmse100, rmse010, rmse001):

        """Obtain public scores for Probe000, Probe100, Probe010, Probe001

        Public,Private

        Probe000,1.250111,1.236582

        Probe100,1.23528,1.221182

        Probe010,1.38637,1.373707

        Probe001,1.29326,1.279869

        """



        def calc(rmse000, n, rmse_i, ni):

            u=(1-(rmse_i**2-rmse000**2)*n/ni)/2

            return u



        u0=calc(rmse000, self.n, rmse100, self.n0)

        u1=calc(rmse000, self.n, rmse010, self.n1)

        u2=calc(rmse000, self.n, rmse001, self.n2)

        u=(self.n0*u0+self.n1*u1+self.n2*u2)/self.n

        return(u0, u1, u2, u)



    def true_means(self):

        # computed by leader board probing

        # u0, u1, u2 and overall mean u

        # Kaggle public scores and Coursera scores slightly differ

        # Kaggle scores

        #return [0.7590957299173547, 0.060230457160248385, 0.39458181098366407, 0.2839717256500001]

        # use Coursera scores here

        return [0.758939742420249, 0.0601995732152425, 0.3945593622881204, 0.28393632703149974]



    def mean_scale(self, filename):

        """Compare the mean of each test group to their true public leaderboard means

        shift the prediction so that the means match

        filename: your submission csv file name

        """

        df=pd.read_csv(filename)

        df.sort_values('ID', ascending=True, inplace=True)

        mask0=self.test.test_group==0

        mask1=self.test.test_group==1

        mask2=self.test.test_group==2

        U=self.true_means()

        print("Group0: predict mean=", df[ mask0 ].item_cnt_month.mean(), "true mean=", U[0])

        print("Group1: predict mean=", df[ mask1 ].item_cnt_month.mean(), "true mean=", U[1])

        print("Group2: predict mean=", df[ mask2 ].item_cnt_month.mean(), "true mean=", U[2])

        change=999

        previous=df.item_cnt_month.values.copy()

        i=1

        while change>1e-6:

            df.loc[mask0, 'item_cnt_month']+=U[0]-df[ mask0 ].item_cnt_month.mean()

            df.loc[mask1, 'item_cnt_month']+=U[1]-df[ mask1 ].item_cnt_month.mean()

            df.loc[mask2, 'item_cnt_month']+=U[2]-df[ mask2 ].item_cnt_month.mean()

            df['item_cnt_month']=df['item_cnt_month'].clip(0,20)

            change=np.sum(np.abs(df.item_cnt_month.values - previous))

            previous=df.item_cnt_month.values.copy()

            print(">loop", i, "change:", change)

            i+=1

        self.save(df, filename.replace('.csv', '_mean.csv'))



    def variance_scale(self, filename, rmse, rmse000=1.250111):

        """

        filename: your submission csv file name

        rmse: your public leaderboard score

        """

        df=pd.read_csv(filename)

        df.sort_values('ID', ascending=True, inplace=True)

        n=df.shape[0]

        u=self.true_means()[-1]

        Yp=df.item_cnt_month.values

        YpYp=np.sum(Yp*Yp)

        YYp=n*(rmse000**2-rmse**2)/2+YpYp/2

        lambda_ = (YYp-u*u*n)/(YpYp-u*u*n)

        print(">>>>>multipler lambda=", lambda_)

        df['item_cnt_month']=(Yp-u)*lambda_+u

        filename2=filename.replace('.csv', '_lambda.csv')

        self.save(df, filename2)

        self.mean_scale(filename2)



    def save(self, df, filename):

        """Produce LeaderBoardProbing file based on dataframe"""

        df = df[['ID','item_cnt_month']].copy()

        df.sort_values(['ID'], ascending=True, inplace=True)

        df['item_cnt_month']=df['item_cnt_month'].apply(lambda x: "%.5f" % x)

        if np.isnan(df.item_cnt_month.isnull().sum()):

            print("ERROR>>>>> There should be no nan entry in the LeaderBoardProbing file!")

        print("Save LeaderBoardProbing to file:", filename)

        df.to_csv(filename, index=False)



    def flip_signs(self, filename):

        """

        Produce LeaderBoardProbing file, flip the sign of prediction for each of the three groups

        filename: your submission csv file name

        output:

            three new submission files with suffix _mpp.csv, _pmp.csv, _ppm.csv

            notation in the notebook

            m: minus, p: plus

            mpp is -++, pmp is +-+, ppm is ++-



        You need to submit these three files to obtain

            rmse_mpp, rmse_pmp, rmse_ppm

        Then you call

            LeaderBoardProbing.variance_scale_v2(filename, rmse_mpp, rmse_pmp, rmse_ppm, rmse)

            Note: rmse is the original rmse score obtained by your filename

        """

        df=pd.read_csv(filename)

        df.sort_values(['ID'], ascending=True, inplace=True)

        mask0=self.test.test_group==0

        mask1=self.test.test_group==1

        mask2=self.test.test_group==2

        tmp=df.copy()

        tmp.loc[mask0, 'item_cnt_month']=-tmp[ mask0 ].item_cnt_month

        self.save(tmp, filename.replace('.csv', '_mpp.csv'))

        tmp=df.copy()

        tmp.loc[mask1, 'item_cnt_month']=-tmp[ mask1 ].item_cnt_month

        self.save(tmp, filename.replace('.csv', '_pmp.csv'))

        tmp=df.copy()

        tmp.loc[mask2, 'item_cnt_month']=-tmp[ mask2 ].item_cnt_month

        self.save(tmp, filename.replace('.csv', '_ppm.csv'))



    def variance_scale_v2(self, filename, rmse_mpp, rmse_pmp, rmse_ppm, rmse):

        """

        filename: your submission csv file name

        You must use LeaderBoardProbing.flip_signs(filename)

            to generate three additional submission files, obtain their public scores

            and feed those scores as parameters

        Scores: rmse-++, rmse+-+, rmse++-, rmse+++



        output:

            New scaled submission file

        """

        df=pd.read_csv(filename)

        df.sort_values(['ID'], ascending=True, inplace=True)

        mask0=self.test.test_group==0

        mask1=self.test.test_group==1

        mask2=self.test.test_group==2

        n=len(df)

        n0=sum(mask0)

        n1=sum(mask1)

        n2=sum(mask2)

        YYp0=n/4*(rmse_mpp**2-rmse**2)

        YYp1=n/4*(rmse_pmp**2-rmse**2)

        YYp2=n/4*(rmse_ppm**2-rmse**2)

        U=self.true_means()

        Yp0=df.loc[mask0, 'item_cnt_month'].values

        Yp1=df.loc[mask1, 'item_cnt_month'].values

        Yp2=df.loc[mask2, 'item_cnt_month'].values

        lambda0=(YYp0-U[0]**2*n0)/(np.sum(Yp0*Yp0)-U[0]**2*n0)

        lambda1=(YYp1-U[1]**2*n1)/(np.sum(Yp1*Yp1)-U[1]**2*n1)

        lambda2=(YYp2-U[2]**2*n2)/(np.sum(Yp2*Yp2)-U[2]**2*n2)

        print("Labmda: ", lambda0, lambda1, lambda2)

        df.loc[mask0, 'item_cnt_month']=U[0]+lambda0*(df[ mask0 ].item_cnt_month-U[0])

        df.loc[mask1, 'item_cnt_month']=U[1]+lambda1*(df[ mask1 ].item_cnt_month-U[1])

        df.loc[mask2, 'item_cnt_month']=U[2]+lambda2*(df[ mask2 ].item_cnt_month-U[2])

        df['item_cnt_month']=df['item_cnt_month'].clip(0,20)

        fn=filename.replace('.csv', '_labmdaV2.csv')

        self.save(df, fn)

        self.mean_scale(fn)

import shutil

shutil.copy("../input/salescompetitionoctmodel/submit_oct.csv", "submit_oct.csv")

# your submission file is now at submit_oct.csv
lbp = LeaderBoardProbing()

lbp.mean_scale('submit_oct.csv')
lbp.variance_scale('submit_oct_mean.csv', 1.118256)
lbp.flip_signs('submit_oct_mean.csv')
lbp.variance_scale_v2('submit_oct_mean.csv', 1.189455, 1.121041, 2.002616, 1.118256)
df=lbp.test.copy()

np.random.seed(42)

df['item_cnt_month']=np.clip(np.random.randn(len(df))*4+1, 0, 20)

lbp.save(df, 'submit_random.csv')

# Submit and we get

# Your public and private LB scores are: 3.473406 and 3.465503.
# You should always run mean_scale, as it requires not probing.

# I actually always send my prediction through mean_scale and submit the processed file

lbp.mean_scale('submit_random.csv')

# Your score for submit_random_mean.csv

# Your public and private LB scores are: 1.545094 and 1.525234.
lbp.variance_scale('submit_random_mean.csv', 1.545094)

# Your score for submit_random_mean_lambda_mean.csv

# Your public and private LB scores are: 1.200340 and 1.185505.



# If you are willing to do three more probes

lbp.flip_signs('submit_random_mean.csv')

# Your scores are

# submit_random_mean_mpp.csv

# Your public and private LB scores are: 1.604641 and 1.583784.

# submit_random_mean_pmp.csv

# Your public and private LB scores are: 1.547342 and 1.527319.

# submit_random_mean_ppm.csv

# Your public and private LB scores are: 1.645131 and 1.624193

lbp.variance_scale_v2('submit_random_mean.csv', 1.604641, 1.547342, 1.645131, 1.545094)

# submit_random_mean_labmdaV2_mean.csv

# Your public and private LB scores are: 1.199848 and 1.184957.
# Let's first generate submission files to obtain rmse000, rmse100, rmse010, rmse001

lbp.probe_mean()

# submit and obtain public leaderboard scores, then use the next line to obtain the group means

lbp.estimate_mean(1.250111, 1.23528, 1.38637, 1.29326)
