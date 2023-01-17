# Install paperetl project

!pip install git+https://github.com/neuml/paperetl



# Install scispacy model

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_md-0.2.5.tar.gz
import os

import shutil



from paperetl.cord19.execute import Execute as Etl



# Copy study design models locally

os.mkdir("cord19q")

shutil.copy("../input/cord19-study-design/attribute", "cord19q")

shutil.copy("../input/cord19-study-design/design", "cord19q")



# Copy previous articles database locally for predictable performance

shutil.copy("../input/cord-19-etl/cord19q/articles.sqlite", "/tmp")



# Build SQLite database for metadata.csv and json full text files

Etl.run("../input/CORD-19-research-challenge", "cord19q", "cord19q", "../input/cord-19-article-entry-dates/entry-dates.csv", False, "/tmp/articles.sqlite")