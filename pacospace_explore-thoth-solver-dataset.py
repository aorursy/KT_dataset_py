!pip install thoth-lab==0.1.11
import json
import pandas as pd

from pathlib import Path
from thoth.lab import solver
solver_repo_path = Path('/kaggle/input/thoth-solver-dataset-v10/solver/')
solver_reports = []
for solver_document_path in solver_repo_path.iterdir():

    with open(solver_document_path, 'r') as solver_json:
        solver_report = json.load(solver_json)

    solver_reports.append(solver_report)

print(f"Number of solver reports is {len(solver_reports)}")
SOLVER_LATEST_VERSION = '1.5.1'
new_solver_reports = [v for v in solver_reports if v['metadata']['analyzer_version'] == SOLVER_LATEST_VERSION]
solver_report = new_solver_reports[0]
solver_report
pd.DataFrame([solver_report["metadata"]])
solver_reports_metadata = []
for solver_document in solver_reports:
    solver_reports_metadata.append(solver.extract_data_from_solver_metadata(solver_document["metadata"]))
solver_reports_metadata_df = pd.DataFrame(solver_reports_metadata)

solver_reports_metadata_df.head(10)
pd.DataFrame([solver_report["result"]])
solver_reports_extracted_data = []
solver_errors = []
for solver_document in solver_reports:
    solver_report_extracted_data = solver.extract_data_from_solver_metadata(solver_document["metadata"])
    for k, v in solver_document["result"].items():
        solver_report_extracted_data[k] = v
        if k == "errors" and v:
            errors = solver.extract_errors_from_solver_result(v)
            for error in errors:
                solver_errors.append(error)
    
    packages = solver.extract_tree_from_solver_result(solver_document["result"])
    solver_report_extracted_data["packages"] = packages
    solver_reports_extracted_data.append(solver_report_extracted_data)
solver_reports_metadata_df = pd.DataFrame(solver_reports_extracted_data)

solver_reports_metadata_df.head(10)
solver_total_errors_df = pd.DataFrame(solver_errors)

solver_total_errors_df.head(10)