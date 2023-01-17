%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
# import advertools as adv

cx = 'CUSTOM_SEARCH_ENGINE_ID'  # used to get data from Google's custom search engine
key = 'API_KEY'                 
candidates = pd.read_csv('../input/midterm_election_candidates_2018_0921.csv')
print(candidates.shape)
candidates.head(2)
candidates.url.isna().agg(['mean','sum'])
cand_summary = (candidates['party']
                .value_counts()
                .to_frame()
                .assign(perc=candidates['party']
                        .value_counts(normalize=True))
                .assign(cum_perc=lambda df: df['perc'].cumsum()))
print('Top 10 parties\ntotal candidates:', candidates.shape[0])
cand_summary.head(10).style.format({'perc': '{:.1%}', 'cum_perc': '{:.1%}'})
fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(10, 6)
ax.bar(cand_summary.index[:10], cand_summary['party'][:10], color='olive')
ax.set_frame_on(False)
ax.grid()
ax.set_title('US Midterm Elections Candidates Per Party - Top 10', fontsize=18)
ax.tick_params(labelsize=14)
ax.set_xlabel('Party', fontsize=14)
ax.set_ylabel('Number of Candidates', fontsize=14)
plt.show()
cand_summary_state = (candidates['state']
                      .value_counts()
                      .to_frame()
                      .assign(perc=candidates['state']
                              .value_counts(normalize=True))
                      .assign(cum_perc=lambda df: df['perc'].cumsum()))
print('Top 10 states (582 out of 1,230 candidates)')#, cand_summary_state['state'].sum())
cand_summary_state.head(10).style.format({'perc': '{:.1%}', 'cum_perc': '{:.1%}'})
fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(10, 6)
ax.bar(cand_summary_state.index[:10], 
       cand_summary_state['state'][:10], color='olive')
ax.set_frame_on(False)
ax.grid()
ax.set_title('US Midterm Elections Candidates Per State - Top 10', fontsize=18)
ax.tick_params(labelsize=14)
ax.set_xlabel('State', fontsize=14)
ax.set_ylabel('Number of Candidates', fontsize=14)
plt.show()
serp_candidates = pd.read_csv('../input/serp_candidates_oct_21.csv')
print(serp_candidates.shape)
serp_candidates.head(2)
top_domains = (serp_candidates
               .displayLink.str.replace('.*(house.gov)', 'house.gov')
               .value_counts())
top_domains.to_frame()[:25]
fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(8,8)
ax.set_frame_on(False)
ax.barh(top_domains.index.str.replace('www.', '')[:15], top_domains.values[:15])
ax.invert_yaxis()
ax.grid()
ax.tick_params(labelsize=14)
ax.set_xticks(range(0, 1000, 100))
ax.set_title('Top Domains Search Ranking - 2018 Midterm Elections', pad=20, fontsize=20)
ax.text(0.5, 1, 'Searching for 1,230 Candidates\' Names', fontsize=16,
        transform=ax.transAxes, ha='center')
ax.set_xlabel('Number of appearances on SERPs', fontsize=14)
plt.show()
top_df = (serp_candidates
          [serp_candidates['displayLink'].str.replace('.*(house.gov)', 'house.gov')
          .isin(top_domains.index[:15])].copy())
top_df['displayLink'] = top_df['displayLink'].str.replace('.*(house.gov)', 'house.gov')

# similar to top_df, but containing all domains:
all_serp_can = (pd.merge(serp_candidates, candidates, how='left', left_on='searchTerms', 
                         right_on='clean_name')
                .sort_values(['searchTerms', 'rank'])
                .reset_index(drop=True))

top_serp_cand = pd.merge(top_df, candidates, how='left', left_on='searchTerms', right_on='clean_name').sort_values(['searchTerms'])
top_serp_cand.head(1)
(all_serp_can
 .pivot_table('rank', ['displayLink'], aggfunc=['count', 'mean'])
 .reset_index()
 .sort_values([('count', 'rank')], ascending=False)
 .reset_index(drop=True)
 .head(15).style.format({('mean', 'rank'): '{:.2f}'}))
for i, party in enumerate(['DEM', 'REP']):
    fig, ax = plt.subplots(1, 1, facecolor='#ebebeb')
    fig.set_size_inches(12, 8)

    ax.set_frame_on(False)
    ax.scatter((top_serp_cand[top_serp_cand['party']==party]
                   .sort_values('displayLink')['displayLink']
                   .str.replace('www.', '')), 
                  (top_serp_cand[top_serp_cand['party']==party]
                   .sort_values('displayLink')['rank']), 
                  s=850, alpha=0.02, edgecolor='k', lw=2, color='navy' if party == 'DEM' else 'darkred')
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    ax.yaxis.set_ticks(range(1, 11))
    ax.tick_params(labelsize=15, rotation=30, labeltop=True,
                   labelbottom=False, length=8)
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel('Search engine results page rank', fontsize=16)
    ax.set_title('Midterm Election Search Ranking for Candidate Names - ' + party, pad=95, fontsize=24)
    fig.tight_layout()
    plt.show()
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt


for i, state in enumerate(cand_summary_state.index[:5]):
    fig, ax = plt.subplots(facecolor='#ebebeb')
    fig.set_size_inches(12, 8)
    ax.set_frame_on(False)
    ax.scatter((top_serp_cand[top_serp_cand['state']==state]
                   .sort_values('displayLink')['displayLink']
                   .str.replace('www.', '')), 
                  (top_serp_cand[top_serp_cand['state']==state]
                   .sort_values('displayLink')['rank']), 
                  s=850, alpha=0.1, edgecolor='k', lw=2, color='olive')
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    ax.yaxis.set_ticks(range(1, 11))
    ax.tick_params(labelsize=15, rotation=30, labeltop=True,
                   labelbottom=False, length=8)
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel('Search engine results page rank', fontsize=16)
    ax.set_title('Midterm Election Search Ranking for Candidate Names - ' + state, pad=95, fontsize=20)
    fig.tight_layout()
    plt.show()
own_domain = (all_serp_can
 [all_serp_can.url.str.replace('https?://(www.)?|/$', '')
  .eq(all_serp_can.displayLink)][['rank','party']].copy())

own_domain_summary = (own_domain
                      .pivot_table('rank', 'party',
                                   aggfunc=['count', 'mean', 'std'])
                      .sort_values([('count', 'rank')], ascending=False))
own_domain_summary
fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(14, 7)
for party in own_domain['party'].unique():
    ax.scatter('party', 'rank', data=own_domain.query('party == @party'), s=800, alpha=0.1)
    ax.set_frame_on(False)
    ax.grid()
    ax.tick_params(labelsize=13)
    ax.set_yticks(range(1, 11))
    ax.invert_yaxis()
    ax.set_title('Search Results Position for Candidates\' Own Domain', fontsize=18, pad=30)
plt.show()