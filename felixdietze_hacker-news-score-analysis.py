import bq_helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")
hacker_news.head("full") # print the first rows to see if it works
# Counts of submissions, submitters, distinct urls, etc per year.
query = """
SELECT
    extract(YEAR from timestamp) as year,
    count(*) as submissions, 
    count(distinct `by`) as submitters,
    /*count(distinct url) as urls,*/
    sum(score) as total_votes,
    count(*)/count(distinct `by`) subm_per_user,
    avg(score) as votes_per_subm,
    approx_quantiles(score,4) as votes_quartiles,
FROM `bigquery-public-data.hacker_news.full`
WHERE `type` = 'story' AND score IS NOT NULL AND score > 0
GROUP BY year
ORDER BY year DESC
"""

result = hacker_news.query_to_pandas_safe(query)
result.set_index('year', inplace=True)
#print(result.to_markdown(floatfmt=(".0f", ".0f", ".0f", ".0f", ".2f", ".2f")))
result
# Scatterplot of all submissions with time on the x-axis and score on a logarithmic y-axis

query = """
SELECT
    time/(365*24*60*60)+1970 as year,
    score + pow(2,rand())-1 as score, /* exponential jitter to make the lower values easier to interpret */
FROM `bigquery-public-data.hacker_news.full`
WHERE `type` = 'story' AND score IS NOT NULL AND score > 0
ORDER BY time DESC
LIMIT 10000000
"""

result = hacker_news.query_to_pandas_safe(query)
splot = sns.jointplot(data=result, x="year", y="score", marginal_kws=dict(bins=3000, rug=False), s=2, alpha=0.05, linewidth=0, height=8)
splot.ax_joint.set_yscale('log')
fig = splot.fig
fig.savefig("scores_over_time.png", dpi=300) 
#result
# count the submissions for different score intervals. Consecutive intervals are twice as big.
query = """
WITH
    stories AS (
        SELECT *
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    ),
    intervals AS (    
        SELECT
            min(score) as min_score,
            max(score) as max_score,
            COUNT(*) as submissions,
            SUM(score) as total_votes,
        FROM
            stories
        GROUP BY
            ceil(log(score)/log(2))
    ),
    totals AS (
        SELECT
            count(*) AS total,
            sum(score) AS total_score,
        FROM stories
    )
SELECT
    max_score,
    [min_score, max_score] as score_interval,
    submissions,
    submissions / totals.total as subm_fraction,
    (SELECT COUNT(*) FROM stories where score <= max_score) / totals.total as cumulative_subm_fraction,
    total_votes,
    total_votes / totals.total_score as votes_fraction,
FROM
    intervals,
    totals
ORDER BY
    min_score ASC
"""

result = hacker_news.query_to_pandas_safe(query)

melted = pd.melt(result, id_vars='max_score', value_vars=['subm_fraction', 'votes_fraction'], var_name="column", value_name="fraction")
plt.figure(figsize=(8.0, 4.8))
splot = sns.barplot(data=melted, x='max_score', y='fraction', hue='column')
fig = splot.get_figure()
fig.savefig("submission_and_votes_distribution_over_score_intervals.png", dpi=300)

result.set_index('score_interval', inplace=True)
del result['max_score']
#print(result.to_markdown(floatfmt=(".0f", ".0f", ".6f", ".6f", ".0f", ".6f")))
result
# count the number of submissions per url/title combination. Then count how many urls have been submitted N times.
query = """
WITH
    stories AS (
        SELECT *
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    ),
    counts AS (
        SELECT
            COUNT(*) as submission_count,
            max(score) as max_score,
        FROM stories
        GROUP BY url,title
    ),
    totals AS (
        SELECT
            count(*) AS total
        FROM counts
    )
SELECT
    counts.submission_count,
    COUNT(*) as urls,
    COUNT(*) / ANY_VALUE(totals.total) as fraction,
    pow(2,avg(log(counts.max_score)/log(2))) as max_score_log2_avg,
    approx_quantiles(counts.max_score, 2)[OFFSET(1)] as max_score_median,
FROM
     counts,
     totals
GROUP BY counts.submission_count
HAVING urls > 50 /* only show submission counts with enough samples */
ORDER BY counts.submission_count
"""

result = hacker_news.query_to_pandas_safe(query)

melted = pd.melt(result, id_vars='submission_count', value_vars=['max_score_log2_avg', 'max_score_median'], var_name="column", value_name="value")
plt.figure(figsize=(8.0, 4.8))
splot = sns.barplot(data=melted, x='submission_count', y='value', hue='column')

fig = splot.get_figure()
fig.savefig("mean_score_for_different_submission_counts.png", dpi=300) 

result.set_index('submission_count', inplace=True)
#print(result.to_markdown(floatfmt=(".0f", ".0f", ".5f", ".2f", ".2f")))
result
# for every submission count, plot a histogram of scores
max_submission_count= 10

query = f"""
WITH
    stories AS (
        SELECT *
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    )
SELECT
    COUNT(*) as submission_count,
    log(max(score) + (pow(2, rand())-1))/log(10) as log10_max_score,   /* exponential jitter to make the lower values easier to interpret for the kdf */
FROM stories
GROUP BY url, title
HAVING submission_count <= {max_submission_count}
ORDER BY max(time) DESC   /* ensure useful plots with limited data */
LIMIT 10000000
"""

result = hacker_news.query_to_pandas_safe(query)

# overlays multiple distplots
# from https://github.com/mwaskom/seaborn/issues/861#issuecomment-549072489
def distplot_fig(data, x, hue=None, row=None, col=None, legend=True, hist=False, **kwargs):
    """A figure-level distribution plot with support for hue, col, row arguments."""
    bins = kwargs.pop('bins', None)
    if (bins is None) and hist: 
        # Make sure that the groups have equal-sized bins
        bins = np.histogram_bin_edges(data[x].dropna())
    g = sns.FacetGrid(data, hue=hue, row=row, col=col, height=8, palette='colorblind')
    g.map(sns.distplot, x, bins=bins, hist=hist, **kwargs)
    if legend and (hue is not None) and (hue not in [x, row, col]):
        g.add_legend(title=hue) 
    return g  

splot = distplot_fig(data=result, x="log10_max_score", hue="submission_count")
fig = splot.fig
fig.savefig("score_histograms_for_different_submission_counts.png", dpi=300) 


splot
# for the best urls+title combinations, which have been submitted more than 5 times, show the scores for every submission.
query = """
SELECT
    url,
    /*FORMAT("[%s...](%s)", SUBSTR(`title`, 0, 30), url) as url,*/
    COUNT(*) as submissions,
    /*COUNT(distinct `by`) as submitters,*/
    /* FORMAT_DATE("%Y-%m-%d", DATE(TIMESTAMP_SECONDS(MIN(time)))) as first, */
    ARRAY_AGG(score order by time ASC) as scores_by_time,
    /*ARRAY_AGG(EXTRACT(HOUR FROM timestamp) order by time ASC) time_of_day,*/
    /*FORMAT_DATE("%Y-%m-%d", DATE(TIMESTAMP_SECONDS(MAX(time)))) as last, */
    approx_quantiles(score, 4) as quartiles,
    DATE_DIFF(DATE(TIMESTAMP_SECONDS(MAX(time))),DATE(TIMESTAMP_SECONDS(MIN(time))), DAY) as days,
    /*DATE_DIFF(DATE(TIMESTAMP_SECONDS(MAX(time))),DATE(TIMESTAMP_SECONDS(MIN(time))), DAY) / count(*) as avg_day,*/
FROM `bigquery-public-data.hacker_news.full`
WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
GROUP BY url,title
HAVING submissions >= 5 /*AND submitters = 1*/ AND days <= 14
ORDER BY max(score) DESC
LIMIT 30
"""

result = hacker_news.query_to_pandas_safe(query)
result.set_index('url', inplace=True)
#print(result.to_markdown())
result
# for urls with different high scores, how different are the titles?
query = """
WITH
    stories AS (
        SELECT *
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    )
SELECT
    /*url,*/
    min(score) as worst,
    max(score) as best,
    ARRAY_AGG(title ORDER BY score ASC LIMIT 1)[OFFSET(0)] as worst_title,
    ARRAY_AGG(title ORDER BY score DESC LIMIT 1)[OFFSET(0)] as best_title,
FROM stories
GROUP BY url
HAVING
    COUNT(*) >= 2
    AND best_title != worst_title
    AND DATE_DIFF(DATE(TIMESTAMP_SECONDS(MAX(time))),DATE(TIMESTAMP_SECONDS(MIN(time))), DAY) < 14
    AND worst/best < 0.8
    AND worst/best > 0.2
ORDER BY worst DESC
LIMIT 50
"""

result = hacker_news.query_to_pandas_safe(query)
print(result.to_markdown())
result

query = """
WITH
    stories AS (
        SELECT *
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    )
SELECT
    /*url,*/
    ARRAY_AGG(score),
    /*ARRAY_AGG(EXTRACT(HOUR FROM timestamp) order by time ASC) time_of_day,*/
    stddev(EXTRACT(HOUR FROM timestamp)) time_of_day_stddev,
FROM stories
GROUP BY url,title
HAVING
    COUNT(*) >= 3
    AND max(score) > 50
    /*AND DATE_DIFF(DATE(TIMESTAMP_SECONDS(MAX(time))),DATE(TIMESTAMP_SECONDS(MIN(time))), DAY) < 14*/

ORDER BY time_of_day_stddev ASC
LIMIT 20
"""

result = hacker_news.query_to_pandas_safe(query)
#print(result.to_markdown())
result
# count the unique urls for different maximal score intervals. Consecutive intervals are twice as big.
query = """
WITH
    urls AS (
        SELECT
            url,
            max(score) as max_score,
            min(score) as min_score,
            ceil(log(max(score))/log(2)) as score_range,
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
        GROUP BY url,title
        HAVING COUNT(*) > 5
    )
SELECT 
    [min(max_score), max(max_score)] as max_score_interval,
    count(*) as urls,
    approx_quantiles(score, 4) as score_quartiles,
FROM urls
JOIN `bigquery-public-data.hacker_news.full` as submissions
    ON submissions.url = urls.url
WHERE submissions.`type` = 'story' AND submissions.score IS NOT NULL AND submissions.score > 0
GROUP BY score_range
ORDER BY score_range
"""

result = hacker_news.query_to_pandas_safe(query)
result.set_index('max_score_interval', inplace=True)
#print(result.to_markdown(floatfmt=(".0f", ".0f", ".0f")))
result
# scatter plot scores of multiple submissions of the same url against its maximum score
min_submission_count = 1
query = f"""
WITH
    stories AS (
        SELECT *
        FROM `bigquery-public-data.hacker_news.full`
        WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    ),
    best AS (
        SELECT
            url,
            title,
            max(score) as max_score,
        FROM stories
        GROUP BY url,title
        HAVING count(*) > {min_submission_count}
    )
SELECT
    /*best.url,*/
    submission.score + pow(2, rand())-1 as score, /* exponential jitter to make the lower values easier to interpret */
    best.max_score + pow(2, rand())-1 as max_score,
FROM best
JOIN stories as submission
    ON submission.url = best.url AND submission.title = best.title
ORDER BY best.max_score DESC
LIMIT 10000000
"""

result = hacker_news.query_to_pandas_safe(query)
plt.figure(figsize=(8, 8))

splot = sns.scatterplot(data=result, x="max_score", y="score", s=4, alpha=0.1, linewidth=0)
splot.set(xscale='log')
splot.set(yscale='log')

fig = splot.get_figure()
fig.savefig("every_url_scores_against_maxscore.png", dpi=300) 


#result
# plot all submission scores against the time of day
query = """
SELECT
    score + (pow(2,rand())-1) as score, /* exponential jitter to make the lower values easier to interpret */
    EXTRACT(HOUR FROM timestamp)+EXTRACT(MINUTE FROM timestamp)/60 as hour_utc,
FROM `bigquery-public-data.hacker_news.full`
WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
ORDER BY time DESC /* a lower limit skips older submissions */
LIMIT 10000000
"""

result = hacker_news.query_to_pandas_safe(query)
splot = sns.jointplot(data=result, x='hour_utc', y='score', s=2, alpha=0.05, linewidth=0, marginal_kws=dict(bins=3000, rug=False), height=6.4)
#splot = sns.jointplot(data=result, x='hour', y='score', kind="kde", levels=30, height=20)
splot.ax_joint.set_yscale('log')

fig = splot.fig
fig.savefig("scores_over_hour.png", dpi=300) 

splot
# for all urls which have been submitted 5 times, look up the hour of the submission with the highest score. Visualize these hours in a histogram.
query = """
WITH maxed AS (
    SELECT
        url,
        max(score) max_score,
    FROM `bigquery-public-data.hacker_news.full`
    WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
    GROUP BY url,title
    HAVING COUNT(*) >= 5 AND max(score)-min(score) >= 1
    )
SELECT
    EXTRACT(HOUR FROM timestamp)+EXTRACT(MINUTE FROM timestamp)/60 as hour,
FROM
    maxed
/* to access timestamp of submission with max_score: */            
JOIN `bigquery-public-data.hacker_news.full` as raw
    ON raw.url = maxed.url AND raw.score = max_score
LIMIT 10000000
        """

result = hacker_news.query_to_pandas_safe(query)
result
plt.figure(figsize=(6.4, 4.8))
splot = sns.distplot(result, bins=24*6, kde=False)

fig = splot.get_figure()
fig.savefig("histogram_of_best_submission_hours.png", dpi=300) 
query = """
        WITH
            maxed AS (
                SELECT
                    url,
                    max(score) max_score,
                    ceil(log(max(score))/log(2)) as log_max_score,
                FROM `bigquery-public-data.hacker_news.full`
                WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
                GROUP BY url,title
                HAVING COUNT(*) >= 2 AND max(score) > 1
            ),
            totals_per_hour AS (
                SELECT
                    count(*) as submissions,
                EXTRACT(HOUR FROM timestamp) as hour_utc,
                FROM `bigquery-public-data.hacker_news.full`
                WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
                GROUP BY hour_utc
            ),
            totals_per_score AS (
                SELECT
                    count(*) as submissions,
                    ceil(log(score)/log(2)) as log_score,
                FROM `bigquery-public-data.hacker_news.full`
                WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
                GROUP BY log_score
            )
        SELECT
            /*raw.url,*/
            /*raw.score,*/
            EXTRACT(HOUR FROM timestamp) as hour_utc,
            count(*) as counts,
            ANY_VALUE(totals_per_hour.submissions) as total_submissions,
            count(*) / ANY_VALUE(totals_per_hour.submissions) / ANY_VALUE(totals_per_score.submissions) as submission_count_normalized,
            log_max_score,
        FROM maxed
        JOIN `bigquery-public-data.hacker_news.full` as raw
            ON raw.url = maxed.url AND raw.score = max_score /* to access timestamp of submission with max_score*/
        JOIN totals_per_hour
            ON totals_per_hour.hour_utc = EXTRACT(HOUR FROM raw.timestamp)
        JOIN totals_per_score
            ON totals_per_score.log_score = ceil(log(raw.score)/log(2))
        WHERE log_max_score <= 11 /* filter outliers above 2048 upvotes */
        GROUP BY hour_utc, log_max_score
        """

result = hacker_news.query_to_pandas_safe(query)
result
fig, ax = plt.subplots()
fig.set_size_inches(6.4, 4.8)
sns.lineplot(data=result, x='hour_utc', y='submission_count_normalized', hue='log_max_score', palette=sns.cubehelix_palette(11, start=.5, rot=-.75, reverse=True))
#sns.distplot(result, kde=False)
#sns.violinplot(data=result, x='hour', y='max_score')
quantiles=2
query = f"""
        WITH       
            urls as (
                /* urls with array of all scores per row */
                SELECT
                    url,
                    ARRAY_AGG(score order by score) as scores,
                FROM `bigquery-public-data.hacker_news.full`
                WHERE `type` = 'story' AND score IS NOT NULL AND score > 0
                GROUP BY url,title
                HAVING COUNT(*) >= {quantiles+1} AND MAX(score) > MIN(score)
            ),
            submissions as (
                /* every submission of an url with its score and the quartile it belongs to. Minimum scores become 0, median scores 0.5 and maximal scores 1. */
                SELECT
                    raw.url,
                    raw.score,
                    EXTRACT(HOUR FROM raw.timestamp) as hour,
                    ROUND((SELECT COUNT(*) FROM UNNEST(scores) as score WHERE score < raw.score) / (ARRAY_LENGTH(scores)-1) * {quantiles})/{quantiles} as quantile,
                FROM
                    urls
                JOIN `bigquery-public-data.hacker_news.full` as raw
                    ON raw.url = urls.url
                WHERE raw.`type` = 'story' AND raw.score IS NOT NULL AND raw.score > 0
            ),
            hour_quantile_bin as (
                /* all submissions grouped by hour and quantile */
                SELECT
                    hour,
                    quantile,
                    count(*) as submission_count
                FROM submissions
                GROUP BY hour, quantile
            ),
            totals as (
                SELECT
                    count(*) as submissions,
                EXTRACT(HOUR FROM timestamp) as hour,
                FROM `bigquery-public-data.hacker_news.full`
                WHERE `type` = 'story' AND url != '' AND score IS NOT NULL AND score > 0
                GROUP BY hour
            )
        SELECT
            hour_quantile_bin.hour as hour_utc,
            hour_quantile_bin.quantile,
            hour_quantile_bin.submission_count,
            totals.submissions as totals,
            hour_quantile_bin.submission_count / totals.submissions as normalized_submission_count
        FROM hour_quantile_bin
        JOIN totals
            ON totals.hour = hour_quantile_bin.hour
        ORDER BY hour_quantile_bin.hour
        """

result = hacker_news.query_to_pandas_safe(query)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
plt.subplots_adjust(left=0.03, right=0.97)
palette = sns.cubehelix_palette(quantiles+1, start=.5, rot=-.75)
splot=sns.lineplot(data=result, x="hour_utc", y="submission_count", hue="quantile",palette=palette, ax=ax1)
sns.lineplot(data=result, x="hour_utc", y="totals",palette=palette, ax=ax2)
sns.lineplot(data=result, x="hour_utc", y="normalized_submission_count", hue="quantile",palette=palette, ax=ax3)
fig = splot.get_figure()
fig.savefig("quantile_hour_distribution.png") 
