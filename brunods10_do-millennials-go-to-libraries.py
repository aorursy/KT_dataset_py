from IPython.display import display, HTML
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f
from scipy.stats import t
from scipy.stats import ttest_ind
import seaborn as sns

df = pd.read_csv("../input/libraries.csv")
display(HTML(df.head().to_html()))

remove = df.columns[0:10].append(df.columns[109:110].append(df.columns[111:]))
df = df.drop(remove, axis = 1)
# To keep things simple, remove non-numeric columns
df = df.select_dtypes(['number'])
display(HTML(df.head().to_html()))
df_m = df[df.age < 35]
df_not = df[df.age >= 35]
print("There are {} Millennials and {} not".format(len(df_m), len(df_not)))

def get_t_test_df(s_sample, N_sample, s_pop, N_pop):
	"""Calculate degrees of freedom between a sample and population.

	Keyword arguments:
	s_sample -- the standard deviation of the sample
	N_sample -- the number of observations in the sample
	s_pop -- the standard deviation of the population
	N_pop -- the number of observations in the population
	"""

	num = (s_sample**2 / N_sample + s_pop**2 / N_pop)**2
	den = (s_sample**2 / N_sample)**2 / (N_sample - 1) + (s_pop**2 
			/ N_pop)**2 / (N_pop - 1)
	v = num / den 
	return v

def t_crit(
		sample_vec, population_vec, colname, alpha=0.05, err=0.001, 
		num_tails=2):
	"""Calculate t-crit values between sample and population vectors.

	Keyword arguments:
	sample_vec -- a single vector of samples
	population_vec -- a single vector of population samples
	colname -- the column name from where the samples came
	alpha -- significance level (default 0.05)
	err -- error; lower is more accurate, more computation (default 0.001)
	num_tails -- number of tails (can be one- or two-tailed) (default 2)
	"""

	sample_vec = [i for i in sample_vec if not isnan(i)]
	sample_vec = np.array(sample_vec)
	population_vec = [i for i in population_vec if not isnan(i)]
	population_vec = np.array(population_vec)
	if num_tails not in {1, 2}:
		raise ValueError("Test must be one- or two-tailed")	
	else:
		alpha = alpha / num_tails
	if len(sample_vec) > 0 and len(population_vec) > 0:
		s_pop = np.std(population_vec)
		N_pop = len(population_vec)
		s_sample = np.std(sample_vec)
		N_sample = len(sample_vec)
		v = get_t_test_df(s_sample, N_sample, s_pop, N_pop)
		t_crit = t.ppf(1-alpha, v)
		return t_crit
	else:
		print(colname, " is empty")
		return float('NaN')

def F_test(sample_vec, population_vec):
	"""Compare variances of two vectors."""

	sample_variance = np.var(sample_vec, ddof=1)
	population_variance = np.var(population_vec, ddof=1)
	variances = [sample_variance, population_variance]
	F = np.max(variances) / np.min(variances)
	return F

def F_crit(sample_vec, population_vec, alpha=0.05, num_tails=2):
	"""Get F-crit between two vectors."""

	alpha = alpha / num_tails
	df1 = np.count_nonzero(~np.isnan(sample_vec)) - 1
	df2 = np.count_nonzero(~np.isnan(population_vec)) - 1
	f_crit = f.ppf(q=1-alpha, dfn=df1, dfd=df2)
	return f_crit

def hotelling_t2(sample_df, population_df):
	"""Conduct Hotelling T2 test on two DataFrames."""

	mean_sample = sample_df.mean()
	n1 = len(sample_df)
	mean_population = population_df.mean()
	n2 = len(population_df)
	df1 = sample_df.shape[1]
	df2 = n1 + n2 - df1
	diff = mean_sample - mean_population
	diff = diff.values
	S1 = np.cov(sample_df, rowvar=False)
	S2 = np.cov(population_df, rowvar=False)
	Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 -2)
	Sp_inv = np.linalg.inv(Sp)
	T2 = np.matmul(np.matmul(np.transpose(diff), Sp_inv), diff) \
			* n1 * n2 / (n1 + n2)
	return T2, df1, df2

def T2_to_F(T2, df1, df2, alpha=0.05, num_tails=2):
	"""Convert Hotelling T2 test to F test.
	
	Necessary step to determine significant if there is a difference between 
	populations.

	Keyword arguments:
	T2 -- the result from the Hotelling T2 test
	df1 -- the first degree of freedom from the Hotelling T2 test
	df2 -- the second degree of freedom from the Hotelling T2 test
	alpha -- the significance level (default 0.05)
	num_tails -- the number of tails specified for the test (default 2)
	"""

	alpha = alpha / num_tails
	F1 = (df2) / (df1 * (df2 + df1 - 1)) * T2
	F = (df2 - 1) / (df1*(df2+df1-2)) * T2
	f_crit = f.ppf(q=1-alpha, dfn=df1, dfd=df2)
	if F > f_crit:
		sig = True
	else:
		sig = False	
	return F, f_crit, sig

def get_stats(df_test, df_control, alpha=0.05, num_tails=2):
	'''Calculate stats for report.'''

	columns = df_test.columns
	means_vector = df_test.mean()
	standard_deviation_vector = df_test.std()
	standard_deviation_vector_control = df_control.std()
	means_vector_control = df_control.mean()
	f_crit_vector = pd.Series([F_crit(df_test[col], df_control[col], alpha) \
			for col in columns], index = columns)
	t_crit_vector = pd.Series([t_crit(df_test[col], df_control[col], col, \
			alpha) for col in columns], index = columns)
	p_values_vector = pd.Series([ttest_ind(df_test[col], df_control[col], \
			equal_var=False)[1] for col in columns], index = columns)
	t_values_vector = pd.Series([ttest_ind(df_test[col], df_control[col], \
			equal_var=False)[0] for col in columns], index = columns)
	significant_difference_vector = pd.Series([p < alpha if ~np.isnan(p) \
			else float('NaN') for p in p_values_vector], index = columns)
	SE = pd.Series([np.sqrt(np.std(df_test[col])**2 
			/ np.count_nonzero(~np.isnan(df_test[col])) 
			+ np.std(df_control[col])**2 
			/ np.count_nonzero(~np.isnan(df_control[col])))
			if (~df_control[col].isnull().all()
			and ~df_test[col].isnull().all())
			else float('NaN') for col in columns], index = columns)
	LL = pd.Series([np.mean(df_test[col]) - t_crit_vector[col] * SE[col] for \
			col in columns], index = columns)
	LU = pd.Series([np.mean(df_test[col]) + t_crit_vector[col] * SE[col] for \
			col in columns], index = columns)
	stats = [means_vector, means_vector_control, standard_deviation_vector, \
		standard_deviation_vector_control, f_crit_vector, t_crit_vector, \
		p_values_vector, t_values_vector, significant_difference_vector, LL, \
		LU]
	df_stats = pd.concat(stats, axis=1)
	df_stats.columns = ["Sample Means",  "Control Means", \
		"Sample Standard Deviations", "Control Standard Deviation", \
		"F Crit", "t Crit", "P Values", "t Values", "Significant", \
		"Lower Bound", "Upper Bound"]
	return df_stats
# Get the Hotelling's t2 stat and degrees of freedom 1 and 2
T2, df1, df2 = hotelling_t2(df_m.drop("age", axis=1), df_not.drop("age", axis=1))

# Input results into T2_to_F to transform the Hotelling's t2 into the F statistic 
# This is done to determine if populations are significantly different based on 95% confidence
F, f_crit, sig = T2_to_F(T2, df1, df2)
print("Hotelling: {}".format(T2))
print("F statistic: {}".format(F))
print("F crit: {}".format(f_crit))
if sig:
    print("The populations are significantly different!")
else:
    print("The populations are not significantly different")
df_stats = get_stats(df_m, df_not)
display(HTML(df_stats.to_html()))
# filter
filt = list(df_stats.Significant == True)
def population_heatmap(df1, df2, name1, name2, save_path=None, filename=None):
	df_dict = {name1:df1, name2:df2}
	for file, df in df_dict.items():
		cols = ['{}...'.format(name[0:10]) if len(name) >= 10 else name \
	   for name in list(df_dict[file].columns)]
		data = df.corr().values
		np.fill_diagonal(data, 0)
		plt.figure(figsize=(10, 10))
		mask = np.zeros_like(data, dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True
		sns.heatmap(data, cmap='coolwarm', cbar=True, yticklabels=cols,\
			mask=mask, xticklabels=cols, vmin=-1, vmax=1, square=True)
		plt.title('Correlations for {}'.format(file.split('.')[0]))
		plt.yticks(rotation=0)
		plt.xticks(rotation=90)
		plt.tight_layout()
		if save_path is not None:
			if filename is not None:
				file = filename
				plt.savefig('{}/{}.png'.format(save_path, file))
			else:
				plt.savefig('{}/Heatmap_{}.png'.format(save_path, file))
			plt.close()
		else:
			plt.show()

def radar_chart(df1, df2, name1, name2, compare=False, save_path=None, filename=None):
	'''
	If compare=True, all plots on same figure
	'''
	df_dict = {name1:df1, name2:df2}
	if compare == True:
		fig = plt.figure(figsize=(20, 20))
	for file, df in df_dict.items():
		cols = ['{}...'.format(name[0:10]) if len(name) >= 10 else name \
	   for name in list(df.columns)]
		data = df.mean()
		N = len(cols)
		angles = [n / float(N) * 2 * np.pi for n in range(N)]
		angles += angles[:1]
		if compare != True:
			fig = plt.figure(figsize=(20, 20))
		ax = plt.subplot(111, polar=True)
		ax.set_theta_offset(np.pi / 2)
		ax.set_theta_direction(-1)		
		plt.xticks(angles[:-1], cols)		
		ax.set_rlabel_position(0)
		ylim = (0, int(data.max()))
		step = 1 # (ylim[1] - ylim[0]) / 4
		plt.yticks(np.arange(ylim[0], ylim[1], step))
		plt.ylim(ylim[0], ylim[1])
		values = data.values.tolist()
		values += values[:1]
		ax.plot(angles, values, linewidth=1, linestyle='solid', label=file)
		ax.fill(angles, values, 'r', alpha=0.1)
		plt.legend(loc='upper right', bbox_to_anchor=(1, 1), \
		  bbox_transform=plt.gcf().transFigure)
		if save_path is not None and compare == False:
			file = file.split('.')[0]
			plt.savefig('{}/Radarchart_{}.png'.format(save_path, file))

	if save_path is not None and compare == True:
		if filename is not None:
			file = filename
			plt.savefig('{}/{}.png'.format(save_path, file))
		else:
			file = str(file_list[0]) + '_' +  str(file_list[1]) + '...'
			plt.savefig('{}/Radarchart_{}.png'.format(save_path, file))
		plt.close()
	else:
		plt.show()

		

def compare_distributions(df1, df2, name_sample, name_control, save_path=None, \
						  filename=None):
	'''
	compares variable distributions between sample and control
	'''
	df1
	df2
	variables = df1.columns
	for variable in variables:
		plt.figure(figsize=(10, 7))
		ax1 = plt.gca()
		ax2 = ax1.twinx()
		s = df1[variable]
		c = df2[variable]
		ax1.hist(s, bins=np.arange(int(min(s))-1.5, int(max(s))+1.5, 1), \
		   alpha=0.25, edgecolor='K', density=True)
		sns.kdeplot(s, label=name_sample)
		ax1.axvline(s.mean(), color='royalblue', linestyle='dashed', \
			  linewidth=2, alpha=0.5)
		ax1.hist(c, bins=np.arange(int(min(c))-1.5, int(max(c))+1.5, 1), \
		   alpha=0.25, edgecolor='K', density=True)
		sns.kdeplot(c, label=name_control)
		ax1.axvline(c.mean(), color='coral', linestyle='dashed', \
			  linewidth=2, alpha=0.5)
		ax1.set_xlabel(variable)
		ax1.set_ylabel("Count")
		ax2.set_ylabel("Probability Density")
		ax2.set_ylim((0, ax2.get_ylim()[1] + 0.15))
		plt.title("Frequency and Probability Distributions of {}".format(variable))
		if save_path is not None:
			if filename is not None:
				plt.savefig('{}/{}.png'.format(save_path, filename))
			else:
				plt.savefig('{}/Distributions_{}_{}_{}.png'.format(save_path,\
				   name_sample, name_control, variable[0:6]))
			plt.close()
		else:
			plt.show()
radar_chart(df_m.iloc[:, filt].drop("age", axis=1), df_not.iloc[:, filt].drop("age", axis=1), "Millennial", "Not", compare=True)
population_heatmap(df_m.iloc[:, filt], df_not.iloc[:, filt], "Millennial", "Not")
compare_distributions(df_m.iloc[:, filt], df_not.iloc[:, filt], "Millennial", "Not")