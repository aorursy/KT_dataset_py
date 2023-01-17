# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install ktrain
from ktrain import text

ts = text.TransformerSummarizer()
sample_doc = """n probability theory and statistics, the negative \

            binomial distribution is a discrete probability \

            distribution that models the number of successes \

            in a sequence of independent and identically distributed \

            Bernoulli trials before a specified (non-random) number \

            of failures (denoted r) occurs. For example, we can \

            define rolling a 6 on a die as a failure, and rolling \

            any other number as a success, and ask how many successful \

            rolls will occur before we see the third failure (r = 3). \

            In such a case, the probability distribution of the number \

            of non-6s that appear will be a negative binomial \

            distribution.The Pascal distribution (after Blaise Pascal) \

            and Polya distribution (for George Pólya) are special cases \

            of the negative binomial distribution. A convention among \

            engineers, climatologists, and others is to use \

            "negative binomial" or "Pascal" for the case of an \

            integer-valued stopping-time parameter r, and use "Polya" \

            for the real-valued case. For occurrences of associated \

            discrete events, like tornado outbreaks, the Polya \

            distributions can be used to give more accurate models \

            than the Poisson distribution by allowing the mean and \

            variance to be different, unlike the Poisson. The negative \

            binomial distribution has a variance \

            {\displaystyle \mu (1+\mu /r)}{\displaystyle \mu (1+\mu /r)}, \

            with the distribution becoming identical to Poisson in the \

            limit {\displaystyle r\to \infty }{\displaystyle r\to\infty}\

            for a given mean {\displaystyle \mu }\mu . This can make \

            the distribution a useful overdispersed alternative to \

            the Poisson distribution, for example for a robust modification \

            of Poisson regression. In epidemiology it has been used to model \

            disease transmission for infectious diseases where the likely \

            number of onward infections may vary considerably from individual \

            to individual and from setting to setting.[2] More generally it \

            may be appropriate where events have positively correlated \

            occurrences causing a larger variance than if the occurrences \

            were independent, due to a positive covariance term.

            

            Suppose there is a sequence of independent Bernoulli trials.

            Thus, each trial has two potential outcomes called "success" 

            and "failure". In each trial the probability of success is p 

            and of failure is (1 − p). We are observing this sequence until

            a predefined number r of successes have occurred. Then the

            random number of failures we have seen, X, will have the negative

            binomial (or Pascal) distribution:

                {\displaystyle X\sim \operatorname {NB} (r,p)}{\displaystyle X\sim \operatorname {NB} (r,p)}

            

            When applied to real-world problems, outcomes of success and

            failure may or may not be outcomes we ordinarily view as good

            and bad, respectively. Suppose we used the negative binomial 

            distribution to model the number of days a certain machine works

            before it breaks down. In this case "failure" would be the result

            on a day when the machine worked properly, whereas a breakdown 

            would be a "success". If we used the negative binomial

            distribution to model the number of goal attempts an athlete 

            makes before scoring r goals, though, then each unsuccessful 

            attempt would be a "failure", and scoring a goal would be 

            "success". If we are tossing a coin, then the negative binomial

            distribution can give the number of tails ("failures") we are 

            likely to encounter before we encounter a certain number of 

            heads ("successes"). In the probability mass function below, 

            p is the probability of success, and (1 − p) is the probability

            of failure.

            """
# Now, let's use our TransformerSummarizer instance to summarize the long document.



ts.summarize(sample_doc)