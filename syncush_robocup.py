import matplotlib
import  pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
df1 =  pd.read_csv('../input/robo2d-18_09_2018.csv')
df_r =  df1[df1['is_source']==True]
df_r = df_r.rename(index=str,columns={'new_dir':'folder'})
df_r.sort_values('year').plot.barh(x='folder',y='analizo_accm_mean',figsize=(7,30))
df_r.groupby(['year','source_lang']).agg(['count','mean'])['analizo_accm_mean']
df_r.groupby(['competition','year','source_lang']).agg(['count','mean']).sort_values(['competition','year'])['analizo_accm_mean']
df_bin_code = df_r[df_r['player_function_cc_radare2_mean'].notnull()]
df_bin_code.plot.scatter(x='player_function_cc_radare2_mean',y='analizo_accm_mean')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os
import re
import random
df_data = pd.read_csv('../input/-02_10_2018.csv')
df_data = df_data[(rep_df_2d['is_source']==True)& (rep_df_2d['binary_player_number_lines']>0) & \
                   ( rep_df_2d['analizo_total_loc']>5000 )]
df_data.head(5)
df_data.tail(5)
df_data.describe()
df_data.isna().sum()
for x in df_data.columns:
    print(x)


cols_corr = [
    "McCabes_cyclomatic_complexity",
    "McCabes_cyclomatic_complexity_per_line_of_comment",
    "McCabes_cyclomatic_complexity_per_module",
    "player_fast_block_instructions_info_max",
    "player_fast_block_instructions_info_std",
    "player_fast_block_instructions_info_sum",
    "player_fast_blocks_sizes_info_max",
    "player_fast_blocks_sizes_info_std",
    "player_fast_blocks_sizes_info_sum",
    "player_fast_complex_mccabe",
    "player_fast_number_of_CFG_nodes_that_return_a_value",
    "player_fast_simple_mccabe",
    "player_function_cc_radare2_max",
    "player_function_cc_radare2_mean",
    "player_function_cc_radare2_std",
    "player_function_cc_radare2_sum",
    "lines_of_code",
    "lines_of_code_per_line_of_comment",
    "lines_of_code_per_module",
    "lines_of_comment",
    "lines_of_comment_per_module",
    "analizo_sc_sum",
    "analizo_sc_variance",
    "analizo_total_abstract_classes",
    "analizo_total_cof",
    "analizo_total_eloc",
    "analizo_total_loc",
    "analizo_total_methods_per_abstract_class",
    "analizo_total_modules",
    "analizo_total_modules_with_defined_attributes",
    "analizo_total_modules_with_defined_methods",
    "analizo_accm_mean"]
corr = df_data[cols_corr].corr()
f, ax = plt.subplots(figsize=(25,25))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True,  ax=ax, vmin=-1, vmax=1, cmap="YlGnBu", annot=True)
ax = sns.regplot(x="lines_of_code", y="player_function_cc_radare2_std", marker="+", data=df_data)
ax = sns.scatterplot(x="analizo_accm_mean", y="player_function_cc_radare2_mean",data=df_data, palette='Reds')
df_data.groupby(by="year")["McCabes_cyclomatic_complexity"].mean().plot(kind="bar")
temp = df_data[df_data["binary_player_number_of_libraries"] == 0]

cols_corr = [
    "McCabes_cyclomatic_complexity",
    "McCabes_cyclomatic_complexity_per_line_of_comment",
    "McCabes_cyclomatic_complexity_per_module",
    "player_fast_block_instructions_info_max",
    "player_fast_block_instructions_info_std",
    "player_fast_block_instructions_info_sum",
    "player_fast_blocks_sizes_info_max",
    "player_fast_blocks_sizes_info_std",
    "player_fast_blocks_sizes_info_sum",
    "player_fast_complex_mccabe",
    "player_fast_number_of_CFG_nodes_that_return_a_value",
    "player_fast_simple_mccabe",
    "player_function_cc_radare2_max",
    "player_function_cc_radare2_mean",
    "player_function_cc_radare2_std",
    "player_function_cc_radare2_sum",
    "lines_of_code",
    "lines_of_code_per_line_of_comment",
    "lines_of_code_per_module",
    "lines_of_comment",
    "lines_of_comment_per_module",
    "analizo_sc_sum",
    "analizo_sc_variance",
    "analizo_total_abstract_classes",
    "analizo_total_cof",
    "analizo_total_eloc",
    "analizo_total_loc",
    "analizo_total_methods_per_abstract_class",
    "analizo_total_modules",
    "analizo_total_modules_with_defined_attributes",
    "analizo_total_modules_with_defined_methods",
    "analizo_accm_mean"]
corr = temp[cols_corr].corr()
f, ax = plt.subplots(figsize=(25,25))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True,  ax=ax, vmin=-1, vmax=1, cmap="YlGnBu", annot=True)
p = temp[cols_corr].corr() - df_data[cols_corr].corr()


cols_corr = [
    "McCabes_cyclomatic_complexity",
    "McCabes_cyclomatic_complexity_per_line_of_comment",
    "McCabes_cyclomatic_complexity_per_module",
    "player_fast_block_instructions_info_max",
    "player_fast_block_instructions_info_std",
    "player_fast_block_instructions_info_sum",
    "player_fast_blocks_sizes_info_max",
    "player_fast_blocks_sizes_info_std",
    "player_fast_blocks_sizes_info_sum",
    "player_fast_complex_mccabe",
    "player_fast_number_of_CFG_nodes_that_return_a_value",
    "player_fast_simple_mccabe",
    "player_function_cc_radare2_max",
    "player_function_cc_radare2_mean",
    "player_function_cc_radare2_std",
    "player_function_cc_radare2_sum",
    "lines_of_code",
    "lines_of_code_per_line_of_comment",
    "lines_of_code_per_module",
    "lines_of_comment",
    "lines_of_comment_per_module",
    "analizo_sc_sum",
    "analizo_sc_variance",
    "analizo_total_abstract_classes",
    "analizo_total_cof",
    "analizo_total_eloc",
    "analizo_total_loc",
    "analizo_total_methods_per_abstract_class",
    "analizo_total_modules",
    "analizo_total_modules_with_defined_attributes",
    "analizo_total_modules_with_defined_methods",
    "analizo_accm_mean"]
corr = p
f, ax = plt.subplots(figsize=(25,25))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True,  ax=ax, vmin=-1, vmax=1, cmap="YlGnBu", annot=True)

