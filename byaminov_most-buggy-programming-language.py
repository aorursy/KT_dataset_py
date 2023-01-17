import pandas as pd
import bq_helper
github = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='github_repos')
query = """
WITH file_changes AS (
    SELECT
        diff.new_path AS file_path,
        commit.commit AS commit_sha,
        -- mark a changed file as 'is_fix' when the commit message looks like a fix of
        -- some problem: contains a full word 'fix', 'fixing', 'fixes' or 'bug', case insensitive
        CASE
            WHEN REGEXP_CONTAINS(commit.message, r"(?i)(^|[^\w])bug([^\w]|$)") THEN 1
            WHEN REGEXP_CONTAINS(commit.message, r"(?i)(^|[^\w])fix(ed|ing|es)?([^\w]|$)") THEN 1
            ELSE 0
        END AS is_fix
    FROM `bigquery-public-data.github_repos.commits` commit
    JOIN UNNEST(commit.difference) diff
    -- exclude merge commits, as they aggregate changes from multiple commits,
    -- and often contain all those commits' messages
    WHERE ARRAY_LENGTH(commit.parent) <= 1)
SELECT
    -- extract the file path part after the last dot, with limited
    -- length, so that it's more probable to be a real file extension
    LOWER(REGEXP_EXTRACT(file_path, r"\.(\w{1,15})$")) AS file_extension,
    COUNT(is_fix) AS n_total_changes,
    SUM(is_fix) AS n_fix_changes,
    COUNT(DISTINCT(commit_sha)) AS n_commits
FROM file_changes
GROUP BY file_extension
HAVING
    -- include only popular file extensions: with many files and many changes in GitHub
    n_total_changes > 10000
    AND n_commits > 10000
-- take the most buggy file extensions
ORDER BY n_fix_changes / n_total_changes DESC
LIMIT 2000
"""
print('Estimated query size: %.1f GB' % github.estimate_query_size(query))
def query_with_custom_timeout(bq_helper_instance, query, timeout_seconds):
    query_job = bq_helper_instance.client.query(query)
    rows = list(query_job.result(timeout=timeout_seconds))
    if len(rows) > 0:
        return pd.DataFrame(data=[list(x.values()) for x in rows],
                            columns=list(rows[0].keys()))
    else:
        return pd.DataFrame()
%%time
file_extension_stats = query_with_custom_timeout(github, query, timeout_seconds=3600)
def set_fixes_ratio(df):
    df['fixes_ratio'] = 100 * df.n_fix_changes / df.n_total_changes
set_fixes_ratio(file_extension_stats)
file_extension_stats.head(20)
popular_languages = {
    'Assembly': '.asm',
    'C or Objective-C': '.c .h .mm',
    'C#': '.cs',
    'C++': '.cc .cpp .cxx .c++ .hh .hpp .hxx .h++',
    'Clojure': '.clj .cljs .cljc .edn',
    'CoffeeScript': '.coffee .litcoffee',
    'Common Lisp': '.lisp .lsp .l .cl .fasl',
    'Dart': '.dart',
    'Elixir': '.ex .exs',
    'Erlang': '.erl .hrl',
    'F#': '.fs .fsi .fsx .fsscript',
    'Go': '.go',
    'Groovy': '.groovy',
    'Haskell': '.hs .lhs',
    'Java': '.java',
    'JavaScript': '.js',
    'Julia': '.jl',
    'Lua': '.lua',
    'Matlab': '.m',
    'Perl': '.pl .pm .t .pod',
    'PHP': '.php .phtml .php3 .php4 .php5 .php7 .phps .php-s',
    'Python': '.py .pyc .pyd .pyo .pyw .pyz ',
    'R': '.r .RData .rds .rda',
    'Ruby': '.rb',
    'Rust': '.rs .rlib',
    'Scala': '.scala .sc',
    'Smalltalk': '.st',
    'SQL': '.sql',
    'Swift': '.swift',
    'TypeScript': '.ts .tsx',
    'VB.NET': '.vb',
    'VBA': '.vba',
    'Shell': '.sh',
    'HTML': '.html',
    'CSS': '.css',
    'Text': '.txt .md .markdown',
    'Kotlin': '.kt'
}
popular_languages_df = pd.DataFrame(
    [(lang, extension.replace('.', '').strip().lower())
     for (lang, extensions) in popular_languages.items()
     for extension in extensions.split()],
    columns=['language', 'file_extension']
)
popular_languages_df.describe()
languages_stats = file_extension_stats\
    .merge(popular_languages_df, left_on='file_extension', right_on='file_extension')\
    .groupby('language')\
    .sum()
# the fixes_ratio column has to be recalculated now based on summed up n_total_changes and n_fix_changes
set_fixes_ratio(languages_stats)
languages_stats = languages_stats.reset_index().sort_values('fixes_ratio', ascending=False)
languages_stats.index = range(len(languages_stats.index))
languages_stats
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=languages_stats,
            x="fixes_ratio", y="language",
            palette="Reds_r")
ax.set(ylabel="",
       xlabel="Percentage of commits that fix something")
sns.despine(left=True, bottom=True)
f, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=languages_stats.sort_values('n_commits', ascending=False),
            x="n_commits", y="language",
            palette="Blues_r")
ax.set(ylabel="",
       xlabel="Number of commits")
sns.despine(left=True, bottom=True)