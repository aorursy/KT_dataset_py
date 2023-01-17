import numpy as np
import pandas as pd
import seaborn as sns

from scipy.optimize import curve_fit
#OUT OF STOCK CLASS
#Detects out of stock days and pedicts probability of out of stock for inference

class out_of_stock_zeroes():
    def __init__(self,sales_array,calculate_oos_array=True):
        self.sales_array = sales_array
        #When do the sales start. A series of zeros at the beginning only means that the product is not yet available and should not be considered as an indicator of out of stock
        start = np.argmax(sales_array != 0)
        sales_array_without_start = sales_array[start:].copy()
        sales_array_start = sales_array[:start].copy()
        
        
        self.counter = self._build_zero_seq_counter(sales_array_without_start)
        
        self.zeroes_seq = self.rle(self.counter,0)

        
        #If we want to calculate an array of same size than self.sales_array with a boolean flag if out of stock
        if calculate_oos_array:
            #First we calculate the minimal sequence length from which we consider that those zeros are due to out of stocl
            self.divisor = self._log_divisor(sales_array) #The higher the mean value of sales, the lower the zeros sequence length needs to be to be considered oos
            self.out_of_stock_min_length = self._get_min_length_oos(self.zeroes_seq,self.divisor)
            
            #Based on the information of what is the out_of_stock_min_length, we can now calculate the oos flag array
            self.zeroes_seq_sales = self.rle(sales_array_without_start,0)
            self.oos_coordinates = self._build_oos_coordinates(start,self.zeroes_seq_sales,self.out_of_stock_min_length)
            self.oos_array = self._build_oos_array(sales_array.shape[0],self.oos_coordinates)
        
        
        # We calculate what is the minimal sequence length of consecutive zeros starting from which out of stock is likely for prediction
        # We use a divisor of 1 in this case as it resulted in better CV
        self.out_of_stock_min_length_for_oos_pred = self._get_min_length_oos(self.zeroes_seq,1)
        
    def _build_zero_seq_counter(self,sales_array):
        '''
        From an array sales_array, builds an array counting the zeros sequence length
        Such as sales_array = [0,1,0,0,1,0,1,0,0,0,0,1]
        Would result in counter = [2,1,0,1]
        '''
        mask = sales_array!=0
        sales_array = sales_array[np.argmax(mask):]
        self.sales_array_without_begin = sales_array
        l = sales_array==0
        counter = np.diff(np.where(np.concatenate(([l[0]],
                                         l[:-1] != l[1:],
                                         [True])))[0])[::2]
        counter = np.bincount(counter)
        counter = counter[1:]
        return counter
        
    def rle(self, a, value=None):
        """
        From an array, calculates the sequences of same value length and position
        Such as a = [10,5,4,0,0,0,0,1,0,0,0,1] and value = 0
        Would result in [(4,3),(3,7)], each element of the array being (length of sequence, position in initial array)
        """
        ia = np.asarray(a)                # force numpy
        n = len(ia)
        if n==0:
            return np.array([[],[]])
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions

        if value is None:
            return np.concatenate([z.reshape(-1,1),p.reshape(-1,1),ia[i].reshape(-1,1)],axis=1)
        else:
            mask = ia[i] == value
            return np.concatenate([z[mask].reshape(-1,1),p[mask].reshape(-1,1)],axis=1)
        
    def _log_divisor(self,a):
        s = a[a>0]
        if s.shape[0]==0:
            return 1e-3
        mean_no_zeros = a[a>0].mean()
        return np.log1p(mean_no_zeros)+1e-3
    
    def _get_min_length_oos(self,zeroes_seq,divisor=1):
        """
        Idea is to look at the zeros sequence length such as [5,4,3,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1]
        When the first zero sequence in the zero sequence array (a bit confusing here sorry) is longer than what came before, then we consider that position to be the minimum sequence length to be oos
        In this example, the first zero sequence of the zero sequence array is in the 4th position but has size 1
        The second sequence is in 6 position but has length 7. So in case divisor=1, the minimal size for oos is 6 => any sequence of zeros in the sales array with a length greater than 6 is to be considered oos
        """
        if zeroes_seq.shape[1]==0:
            return self.sales_array.shape[0]
        zeroes_seq_tmp = zeroes_seq.copy()
        zeroes_seq[:,0] = zeroes_seq[:,0] + zeroes_seq[:,1] + 1
        zeroes_seq_tmp[:,1] = zeroes_seq_tmp[:,1]/divisor
        filtered_seq = zeroes_seq[zeroes_seq_tmp[:,0]>zeroes_seq_tmp[:,1]]
        if filtered_seq.shape[0]==0:
            return self.sales_array.shape[0]
        return filtered_seq[0][1]+1
    
    def _build_oos_coordinates(self,start,zeroes_seq_sales,out_of_stock_min_length):
        oos_coordinates = zeroes_seq_sales[zeroes_seq_sales[:,0]>=out_of_stock_min_length].copy()
        oos_coordinates[:,1] = oos_coordinates[:,1] + start
        oos_coordinates = np.concatenate([[[start,0]],oos_coordinates],axis=0)
        return oos_coordinates
        
    def _build_oos_array(self,length,oos_coordinates):
        a = np.zeros(length)
        for coord in oos_coordinates:
            a[coord[1]:coord[1]+coord[0]] = 1
        return a
    
    
    def curve_func(self,x,a):
        return 1/(1+a*np.sqrt(x))

    
    def build_oos_prob_array(self,zeros_before,alpha=2):
        """
        This function calculates an array of size 28 that will serve as multiplier array to the inference.
        Idea is to transform any element that we consider oos to zero in the inference
        For example if just before the inference, a product had a sequence of 50 zeros while usually it is never zero, we create an array to transform the inference to zero.
        
        Also for each sales_array sequence, we fit a simple curve (1/(1+a*np.sqrt(x))) to give us the probability that a sequence will have a size N given how long it is already (n_zeros_before) and how long the oos sequences were in the history of this sales array
        For example if a sales_array has oos sequences such as [0,0,0,0,0,4,3,2,0,1] meaning that the shortest oos sequence has size 6 and occured 4 times in the past
        Then we fit x=[1,2,3,4,5] y=[4,3,2,0,1]/sum([4,3,2,0,1]) to get the parameter a
        So now if we have a sequence of zeroes before of size 7, we can plug calculate the probability of oos based on the curve that we just fit
        
        The higher the alpha, the lower the probability of oos sequence
        """
        counter_cumsum = np.cumsum(self.counter)
        #How many times oos sequences occured
        oos_occurences = counter_cumsum[-1] - counter_cumsum[self.out_of_stock_min_length_for_oos_pred-1]
        
        #Min zeroes sequence after which 1/alpha% chances of a oos
        alpha_oos_zeros_seq_size_prob = np.argmax(counter_cumsum>counter_cumsum[-1]-alpha*oos_occurences)
       
    #if zeros before inferior to half_oos_zeros_seq_size_prob then not a oos sequence
        if zeros_before<alpha_oos_zeros_seq_size_prob:
            return np.ones(28)
        
        #Inverse cumsum of zeroes occurences
        inverse_cumsum = counter_cumsum[-1] - counter_cumsum[alpha_oos_zeros_seq_size_prob:]
        
        #Get indices of oos occurences
        x_oos_occurences = np.concatenate([np.array([0]),np.where(inverse_cumsum[1:] != inverse_cumsum[:-1])[0]+1])
        y_oos_occurences = inverse_cumsum[x_oos_occurences]
        y_oos_occurences = y_oos_occurences / y_oos_occurences[0]
        if y_oos_occurences.shape[0]<=2:
            x_oos_occurences = np.insert(x_oos_occurences, -1, x_oos_occurences[-1]-0.1)
            y_oos_occurences = np.insert(y_oos_occurences, -1, 1e-5)
        
        #Fit curve
        a,cor = curve_fit(self.curve_func, x_oos_occurences, y_oos_occurences,p0=[1],bounds=([0],[np.inf]))
        
        #Calculate probs with zeroes before
        x_prob = np.array([zeros_before - alpha_oos_zeros_seq_size_prob + i for i in range(29)])
        y_prob = self.curve_func(x_prob,a[0])
        y_prob = y_prob/y_prob[0]
        
        #Arbitrary values for oos calculation optimized with CV
        oos_new_array = np.where(y_prob>0.4,
                                 np.where(y_prob>0.7,0,1-y_prob),
                                 1)
        return oos_new_array[1:]
sales_pd = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
a = sales_pd.iloc[17,6:].values.astype('int')
oos = out_of_stock_zeroes(a)
#In green, actual sales
sns.lineplot(x=np.arange(a.shape[0]),y=a,color='green')
#In red, out of stock flag. When 1 => out of stock day
sns.lineplot(x=np.arange(a.shape[0]),y=oos.oos_array,color='red')
a_without_last_28_days = sales_pd.iloc[17,6:-28].values.astype('int')
oos_without_last_28_days = out_of_stock_zeroes(a_without_last_28_days)
#out_of_stock_zeroes class has method to "predict" if the next 28 days will be out of stock based on past out of stock behavior
#Method takes as argument the number of consecutive zeros just before the inference
#It outputs an array that can be multiplied with the results of an inference model. If the likelihood of out of stock is high, then the multiplication element will be zero
#If no zeros, then no out of stock
#The more consecutive zeros before the inference, the more likely there will be out of stock in the inference

mask = a_without_last_28_days!=0
n_zeroes_before = np.argmax(mask[::-1])
out_of_stock_multiplier_array = oos_without_last_28_days.build_oos_prob_array(n_zeroes_before)

out_of_stock_multiplier_array
#If less consecutive zeroes before the inference, the probability of out of stock is lower
out_of_stock_multiplier_array = oos_without_last_28_days.build_oos_prob_array(6)

out_of_stock_multiplier_array
