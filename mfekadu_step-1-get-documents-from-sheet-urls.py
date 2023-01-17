!pip install pathvalidate==2.3.0
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import requests

from urllib.parse import urlparse

from pathvalidate import sanitize_filename
!pwd

!echo "====="

!mkdir downloads

!ls
EXCEL_FILE = "/kaggle/input/syllabus-corpus/thruuu.xlsx"

OUTPUT_FOLDER = "/kaggle/working/downloads"

CSV_FILENAME = "downloads.csv"

ARCHIVE_FILENAME = "downloads.tar.gz"



# Set the environment variables within this notebook

# to allow for referencing via in shell commands

# e.g. `!echo $EXCEL_FILE`

os.environ["EXCEL_FILE"] = EXCEL_FILE

os.environ["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.environ["CSV_FILENAME"] = CSV_FILENAME

os.environ["ARCHIVE_FILENAME"] = ARCHIVE_FILENAME
!echo $EXCEL_FILE "\n"

!echo $OUTPUT_FOLDER "\n"

!echo $CSV_FILENAME "\n"

!echo $ARCHIVE_FILENAME "\n"
df = pd.read_excel(EXCEL_FILE, sheet_name="SERP")
df.head()
print(df.shape)
print(len(df["Title"].unique()))

df[df["Title"].duplicated()]
def sanitize(url):

    # extra sanitize because Kaggle does not like "~"

    return (

        sanitize_filename(url)

        .replace("~", "")

        .replace("&", "")

        .replace("=", "")

        .replace("#", "")

        .replace("%", "")

        .replace("$", "")

        .replace("@", "")

    )
sanitize("wowowo&=#owowowow/http://wow.wow.com/~wow/woow")
def _extract_last_path_part(url):

    return sanitize(os.path.split(urlparse(url).path)[1])
_extract_last_path_part("https://www.google.com/wow/ok/wo&$@%w")
for index, row in df.iterrows():

    pos = row["Position"]

    serp_title = row["Title"]

    url = row["URL"]

    # fname = sanitize(url)

    fname = _extract_last_path_part(url)

    fpath = os.path.join(OUTPUT_FOLDER, f"{pos}___{fname}")

    response = requests.get(url)

    print(fpath, " "*50, "\r" ,end="")

    with open(fpath, 'wb') as f:

        f.write(response.content)
!ls downloads | head -5
!ls downloads | wc -l
!xxd downloads/100__* | head -5
print(ARCHIVE_FILENAME)

!echo $ARCHIVE_FILENAME
!tar -cvzf $ARCHIVE_FILENAME downloads/*
# !rm -rf downloads

!ls -lah
df.to_csv(CSV_FILENAME, index=False)
pd.read_csv(CSV_FILENAME).head()
!ls
# ...........................................

# ...........................................

# ...........................................

# !tar -xvzf $ARCHIVE_FILENAME

# downloads/1___LA114-213Lec.pdf

# downloads/2___syllabus.pdf

# downloads/3___syllabus.pdf

# downloads/4___Syllabus-Lean-Sigma-Green-Belt-Cert12.pdf

# downloads/5___syllabus.pdf

# downloads/6___syllabus.pdf

# downloads/7___viewcontent.cgi

# downloads/8___syllabus.pdf

# ...........................................

# ...........................................

# ...........................................

# !ls downloads

# 1___LA114-213Lec.pdf			       5___syllabus.pdf

# 2___syllabus.pdf			       6___syllabus.pdf

# 3___syllabus.pdf			       7___viewcontent.cgi

# 4___Syllabus-Lean-Sigma-Green-Belt-Cert12.pdf  8___syllabus.pdf

# ...........................................

# ...........................................

# ...........................................