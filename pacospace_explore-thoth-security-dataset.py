!pip install thoth-lab==0.2.3
import json
import pandas as pd

from pathlib import Path
from thoth.lab.security import SecurityIndicators
security_indicators = SecurityIndicators()
security_indicator_bandit_repo_path = Path('/kaggle/input/thoth-security-dataset-v10/security/si-bandit/')
security_indicator_bandit_reports = []

for security_indicator_bandit_path in security_indicator_bandit_repo_path.iterdir():

    with open(security_indicator_bandit_path, 'r') as si_bandit_json:
        si_bandit_report = json.load(si_bandit_json)

    security_indicator_bandit_reports.append(si_bandit_report)

print(f"Number of solver reports is {len(security_indicator_bandit_reports)}")
security_indicator_cloc_repo_path = Path('/kaggle/input/thoth-security-dataset-v10/security/si-cloc/')
security_indicator_cloc_reports = []

for security_indicator_cloc_path in security_indicator_cloc_repo_path.iterdir():
    
    with open(security_indicator_cloc_path, 'r') as si_cloc_json:
        si_cloc_report = json.load(si_cloc_json)

    security_indicator_cloc_reports.append(si_cloc_report)

print(f"Number of solver reports is {len(security_indicator_cloc_reports)}")
security_indicator_bandit_report = security_indicator_bandit_reports[0]
metadata_df = security_indicators.create_si_bandit_metadata_dataframe(
    si_bandit_report=security_indicator_bandit_report
)
metadata_df
si_bandit_report_result_metrics_df = pd.DataFrame(security_indicator_bandit_report["result"]['metrics'])
si_bandit_report_result_metrics_df
filename = si_bandit_report_result_metrics_df.columns.values[0]
filename
si_bandit_report_result_metrics_df[filename]
si_bandit_report_result_results_df = pd.DataFrame(security_indicator_bandit_report["result"]['results'])
si_bandit_report_result_results_df
subset_df = si_bandit_report_result_results_df[si_bandit_report_result_results_df["filename"].values == filename]
subset_df
security_confidence_df, summary_files = security_indicators.create_security_confidence_dataframe(
    si_bandit_report=security_indicator_bandit_report
)
security_confidence_df
si_bandit_report_summary_df = security_indicators.produce_si_bandit_report_summary_dataframe(
    metadata_df=metadata_df,
    si_bandit_sec_conf_df=security_confidence_df,
    summary_files=summary_files
    
)
si_bandit_report_summary_df
security_indicator_cloc_report = security_indicator_cloc_reports[0]
metadata_df = security_indicators.create_si_cloc_metadata_dataframe(
    si_cloc_report=security_indicator_cloc_report
)
metadata_df
results_df = security_indicators.create_si_cloc_results_dataframe(si_cloc_report=security_indicator_cloc_report)
results_df
summary_df = security_indicators.produce_si_cloc_report_summary_dataframe(
    metadata_df=metadata_df,
    cloc_results_df=results_df
)
summary_df
FILTER_FILES = ["tests/", "/test"]
final_df = security_indicators.create_si_bandit_final_dataframe(
    si_bandit_reports=security_indicator_bandit_reports,
    use_external_source_data=True,
    filters_files=FILTER_FILES
)
final_df.shape
final_df.drop_duplicates(
    subset=['analyzer_version', 'package_name', "package_version", "package_index"], inplace=True
)
final_df.shape
final_df.head()
final_df.describe()
from thoth.common.helpers import parse_datetime
filter_date = parse_datetime("2018-01-01T00:00:00.000")
filtered_df = final_df[final_df['release_date'] > filter_date]
filtered_df.head()
sorted_df = filtered_df.sort_values(by=['SEVERITY.HIGH__CONFIDENCE.HIGH', 'SEVERITY.HIGH__CONFIDENCE.MEDIUM', 'SEVERITY.MEDIUM__CONFIDENCE.HIGH'], ascending=False)
sorted_df.head()
security_indicators.create_vulnerabilities_plot(
    security_df=sorted_df.head(30)
)
package_summary_df = sorted_df[(sorted_df['package_name'] == "acme") & (sorted_df['package_index'] == "https://pypi.org/simple")]
package_summary_df = package_summary_df.sort_values(by=['release_date'], ascending=True)
security_indicators.create_package_releases_vulnerabilities_trend(
    package_summary_df=package_summary_df,
)
package_summary_df = sorted_df[(sorted_df['package_name'] == "aiida-core") & (sorted_df['package_index'] == "https://pypi.org/simple")]
package_summary_df = package_summary_df.sort_values(by=['release_date'], ascending=True)
security_indicators.create_package_releases_vulnerabilities_trend(
    package_summary_df=package_summary_df
)
package_summary_df = sorted_df[(sorted_df['package_name'] == "aiohttp") & (sorted_df['package_index'] == "https://pypi.org/simple")]
package_summary_df = package_summary_df.sort_values(by=['release_date'], ascending=True)
security_indicators.create_package_releases_vulnerabilities_trend(
    package_summary_df=package_summary_df
)
si_cloc_total_df = security_indicators.create_si_cloc_final_dataframe(
    si_cloc_reports=security_indicator_cloc_reports
)
si_cloc_total_df.shape
si_cloc_total_df.drop_duplicates(
    subset=['analyzer_version', 'package_name', "package_version", "package_index"], inplace=True
)
si_cloc_total_df.shape
si_cloc_total_df.head()
