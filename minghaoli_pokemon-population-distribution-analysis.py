# Analysis
import numpy as np
import pandas as pd
import scipy.stats as sps

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pkmn_df = pd.read_csv("../input/Pokemon.csv")
pkmn_df.info()
# Replace null values with a more sensible string value.
pkmn_df.fillna("None", inplace = True)
pkmn_df.head(20)
pkmn_df.tail(20)
non_special_pkmn = pkmn_df[~(pkmn_df.Name.str.contains("Mega|Primal")) & ~(pkmn_df.Legendary)]
special_pkmn = pkmn_df[~pkmn_df.index.isin(non_special_pkmn.index)]

# T-test
non_special_vs_special = sps.ttest_ind(non_special_pkmn.Total, special_pkmn.Total)

print("Number of \"non-special\" Pokemon: \t%d" %(non_special_pkmn.shape[0]))
print("Number of \"special\" Pokemon: \t\t%d" %(special_pkmn.shape[0]))
print("\nNon-Special vs. Special Total stat independent T-test:")
print("T Value: %.2f" %(non_special_vs_special[0]))
print("P Value: %.2e" %(non_special_vs_special[1]))

non_legend_megas = special_pkmn[~special_pkmn.Legendary]
legendary_pkmn = special_pkmn[special_pkmn.Legendary]

# Hoopa Unbound form provides a stat boost, will be considered as a "mega".
non_mega_legends = special_pkmn[
    ~(special_pkmn.Name.str.contains("Mega|Primal|Unbound")) &
    (special_pkmn.Legendary)
]
mega_legends = special_pkmn[
    ~(special_pkmn.index.isin(non_mega_legends.index)) &
    ~(special_pkmn.index.isin(non_legend_megas.index))
]

# T-tests
non_legend_mega_vs_base_legend = sps.ttest_ind(
    non_legend_megas.Total, non_mega_legends.Total
)
base_legend_vs_mega_legend = sps.ttest_ind(
    non_mega_legends.Total, mega_legends.Total
)

print(
    "Number of non-Legendary Mega Evolutions: \t%d" %(non_legend_megas.shape[0])
)
print("Number of base Legendary Pokemon: \t\t%d" %(non_mega_legends.shape[0]))
print("Number of Legendary Mega Evolutions: \t\t%d" %(mega_legends.shape[0]))
print()
print("Non-Legendary Mega vs. base Legendary Total stat independent T-test:")
print("T Value: %.2f" %(non_legend_mega_vs_base_legend[0]))
print("P Value: %.2f" %(non_legend_mega_vs_base_legend[1]))
print("\nBase Legendary vs Legendary Mega Total stat independent T-test:")
print("T Value: %.2f" %(base_legend_vs_mega_legend[0]))
print("P Value: %.2e" %(base_legend_vs_mega_legend[1]))

kde_figs, (ax_ns_v_s, ax_nlm_v_nml, ax_nl_v_l) = \
    plt.subplots(3, 1, figsize = [10, 15], sharex = True, sharey = True)

sns.kdeplot(non_special_pkmn.Total, ax = ax_ns_v_s, shade = True)
sns.kdeplot(special_pkmn.Total, ax = ax_ns_v_s, shade = True)

sns.kdeplot(
    non_legend_megas.Total, ax = ax_nlm_v_nml, shade = True,
    color = "green"
)
sns.kdeplot(
    non_mega_legends.Total, ax = ax_nlm_v_nml, shade = True,
    color = "red"
)

sns.kdeplot(
    non_special_pkmn.Total, ax = ax_nl_v_l, shade = True,
    color = "purple"
)
sns.kdeplot(
    legendary_pkmn.Total, ax = ax_nl_v_l, shade = True,
    color = "gold"
)
sns.despine()

ax_ns_v_s.set_title("Non-Special vs. Special", fontsize = 16)
ax_ns_v_s.legend(labels = ["Non-Special", "Special"])
ax_nlm_v_nml.set_title(
    "Non-Legendary Megas vs. Base Legendaries", fontsize = 16
)
ax_nlm_v_nml.legend(labels = ["Non-Legendary Mega", "Base Legendary"])
ax_nl_v_l.set_title("Non-Legendary vs. Legendary", fontsize = 16)
ax_nl_v_l.legend(labels = ["Non-Legendary", "Legendary"])
kde_figs.suptitle(
    "Kernel Density Estimate Comparisons of Different Pokemon Subsets",
    fontsize = 20, y = 0.93
)
kde_figs.text(
    x = 0.5, y = 0.08, s = "Total Stat", ha = "center", fontsize = 16
)
kde_figs.text(
    0.04, 0.5, "Proportion", va = "center", rotation = "vertical",
    fontsize = 16
)

# Formatting
type_dist_fig, (ax_pt, ax_st) = plt.subplots(2, 1, figsize = [10, 15], sharex = True)
sns.countplot(
    y = "Type 1", data = pkmn_df,
    order = pkmn_df["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax_pt
)
sns.countplot(
    y = "Type 2", data = pkmn_df,
    order = pkmn_df["Type 2"].value_counts().index,
    palette = "PuBu_d", ax = ax_st
)
sns.despine()

# Labeling
ax_pt.set_title("Primary Type Frequencies", fontsize = 16)
ax_pt.set_xlabel("")
ax_pt.set_ylabel("Primary Type", fontsize = 14)
ax_st.set_title("Secondary Type Frequencies", fontsize = 16)
ax_st.set_xlabel("Count", fontsize = 14)
ax_st.set_ylabel("Secondary Type", fontsize = 14)
type_dist_fig.suptitle("Type Distributions", fontsize = 20, y = 0.93)

primary_type_encoded = pd.get_dummies(pkmn_df["Type 1"])
primary_type_encoded["None"] = 0
secondary_type_encoded = pd.get_dummies(pkmn_df["Type 2"])
type_corr = (primary_type_encoded + secondary_type_encoded).corr()

# Figure
type_heatmap, ax_h = plt.subplots(figsize = [15, 10])
sns.heatmap(type_corr, cmap = "GnBu", linewidth = 0.01, ax = ax_h)
type_heatmap.suptitle(
    "Type Correlation Matrix", fontsize = 20, x = 0.45, y = 0.93
)

# Formatting
gen1_type = pkmn_df[pkmn_df.Generation == 1].iloc[:, 2:4]
gen2_type = pkmn_df[pkmn_df.Generation == 2].iloc[:, 2:4]
gen3_type = pkmn_df[pkmn_df.Generation == 3].iloc[:, 2:4]
gen4_type = pkmn_df[pkmn_df.Generation == 4].iloc[:, 2:4]
gen5_type = pkmn_df[pkmn_df.Generation == 5].iloc[:, 2:4]
gen6_type = pkmn_df[pkmn_df.Generation == 6].iloc[:, 2:4]

type_dist_gen_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = \
    plt.subplots(2, 3, figsize = [30, 20], sharex = True)

sns.countplot(
    y = "Type 1", data = gen1_type,
    order = gen1_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax1
)
sns.countplot(
    y = "Type 1", data = gen2_type,
    order = gen2_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax2
)
sns.countplot(
    y = "Type 1", data = gen3_type,
    order = gen3_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax3
)
sns.countplot(
    y = "Type 1", data = gen4_type,
    order = gen4_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax4
)
sns.countplot(
    y = "Type 1", data = gen5_type,
    order = gen5_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax5
)
sns.countplot(
    y = "Type 1", data = gen6_type,
    order = gen6_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax6
)
sns.despine()

# Labeling
ax1.set_title("Gen 1", fontsize = 20)
ax2.set_title("Gen 2", fontsize = 20)
ax3.set_title("Gen 3", fontsize = 20)
ax4.set_title("Gen 4", fontsize = 20)
ax5.set_title("Gen 5", fontsize = 20)
ax6.set_title("Gen 6", fontsize = 20)

ax1.set_xlabel(""); ax1.set_ylabel("")
ax2.set_xlabel(""); ax2.set_ylabel("")
ax3.set_xlabel(""); ax3.set_ylabel("")
ax4.set_xlabel(""); ax4.set_ylabel("")
ax5.set_xlabel(""); ax5.set_ylabel("")
ax6.set_xlabel(""); ax6.set_ylabel("")

type_dist_gen_fig.suptitle(
    "Generational Type Distributions", fontsize = 30, y = 0.93
)
type_dist_gen_fig.text(
    x = 0.5, y = 0.08, s = "Count", ha = "center", fontsize = 25
)
type_dist_gen_fig.text(
    0.08, 0.5, "Primary Type", va = "center", rotation = "vertical",
    fontsize = 25
)

# Formatting
lgd_type_dist_fig, (ax_lpt, ax_lst) = \
    plt.subplots(2, 1, figsize = [10, 15], sharex = True)
sns.countplot(
    y = "Type 1", data = legendary_pkmn,
    order = legendary_pkmn["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax_lpt
)
sns.countplot(
    y = "Type 2", data = legendary_pkmn,
    order = legendary_pkmn["Type 2"].value_counts().index,
    palette = "PuBu_d", ax = ax_lst
)
sns.despine()

# Labeling
ax_lpt.set_title("Primary Type Frequencies", fontsize = 16)
ax_lpt.set_xlabel("")
ax_lpt.set_ylabel("Primary Type", fontsize = 14)
ax_lst.set_title("Secondary Type Frequencies", fontsize = 16)
ax_lst.set_xlabel("Count", fontsize = 14)
ax_lst.set_ylabel("Secondary Type", fontsize = 14)
lgd_type_dist_fig.suptitle(
    "Legendary Type Distributions", fontsize = 20, y = 0.93
)

g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Total", color = "red", shade = True)
g.fig.suptitle(
    "Total Stat Distribution Across Types", y = 1.01, fontsize = 20
)

g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "HP", color = "green", shade = True)
g.fig.suptitle(
    "HP Distribution Across Types", y = 1.01, fontsize = 20
)
g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Attack", color = "purple", shade = True)
g.fig.suptitle(
    "Attack Distribution Across Types", y = 1.01, fontsize = 20
)
g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Defense", color = "blue", shade = True)
g.fig.suptitle(
    "Defense Distribution Across Types", y = 1.01, fontsize = 20
)
g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Sp. Atk", color = "orange", shade = True)
g.fig.suptitle(
    "Special Attack Distribution Across Types", y = 1.01, fontsize = 20
)
g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Sp. Def", color = "gold", shade = True)
g.fig.suptitle(
    "Special Defense Distribution Across Types", y = 1.01, fontsize = 20
)
g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Speed", color = "brown", shade = True)
g.fig.suptitle(
    "Speed Distribution Across Types", y = 1.01, fontsize = 20
)
stat_dist_gen_fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = \
    plt.subplots(4, 2, figsize = [40, 20], sharex = True)


sns.violinplot(x = "Generation", y = "Total", data = pkmn_df, ax = ax1)
sns.violinplot(x = "Generation", y = "HP", data = pkmn_df, ax = ax2)
sns.violinplot(x = "Generation", y = "Attack", data = pkmn_df, ax = ax3)
sns.violinplot(x = "Generation", y = "Defense", data = pkmn_df, ax = ax4)
sns.violinplot(x = "Generation", y = "Sp. Atk", data = pkmn_df, ax = ax5)
sns.violinplot(x = "Generation", y = "Sp. Def", data = pkmn_df, ax = ax6)
sns.violinplot(x = "Generation", y = "Speed", data = pkmn_df, ax = ax7)

ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

ax1.set_ylabel("Total", fontsize = 18)
ax2.set_ylabel("HP", fontsize = 18)
ax3.set_ylabel("Attack", fontsize = 18)
ax4.set_ylabel("Defense", fontsize = 18)
ax5.set_ylabel("Sp. Atk", fontsize = 18)
ax6.set_ylabel("Sp. Def", fontsize = 18)
ax7.set_ylabel("Speed", fontsize = 18)

stat_dist_gen_fig.suptitle(
    "Generational Stat Distributions", fontsize = 40,
    y = 0.93
)
stat_dist_gen_fig.text(
    x = 0.5, y = 0.08, s = "Generation", ha = "center", fontsize = 30
)
stat_dist_gen_fig.text(
    0.08, 0.5, "Stat", va = "center", rotation = "vertical",
    fontsize = 30
)

sns.despine()