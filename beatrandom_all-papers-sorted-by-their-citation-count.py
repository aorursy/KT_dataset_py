import os

import json

datadir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/'

paperAndCount = {}

for filename in os.listdir(datadir):

	with open(os.path.join(datadir, filename)) as file:

			data=json.load(file)

	if len(data["metadata"]["title"]) != 0:

		if data["metadata"]["title"] not in paperAndCount :

			paperAndCount[data["metadata"]["title"]] = 1

		else :

			oldCount = paperAndCount[data["metadata"]["title"]]

			paperAndCount[data["metadata"]["title"]] = oldCount+1

	for reference in data["bib_entries"].items():

		if (not reference[1]["title"].startswith("Submit your next manuscript to BioMed Central")) and ("This article is an" not in reference[1]["title"]) and (len(reference[1]["title"])) != 0:

			if reference[1]["title"] not in paperAndCount :

				paperAndCount[reference[1]["title"]] = 1

			else :

				oldCount = paperAndCount[reference[1]["title"]]

				paperAndCount[reference[1]["title"]] = oldCount+1

output = open('papersSortedByCitationCount', "a")

for paper,count in sorted(paperAndCount.items(), key=lambda item:item[1], reverse=True)  :

	output.write(paper + "\t" + str(count) + "\n")

output.close()