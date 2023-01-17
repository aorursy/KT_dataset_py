import numpy as np

import pandas as pd

import requests

from lxml import html
package_list_url = 'https://docs.anaconda.com/anaconda/packages/py3.6_linux-64/'

page = requests.get(package_list_url)

tree = html.fromstring(page.content)
rows = tree.xpath('//table[@class="docutils"]//tr')



# Drop the first row containing column titles.

rows.pop(0);
package_names = [row.xpath('td[1]/a/text()')[0] for row in rows]



summaries_and_licenses = [row.xpath('td[3]/text()')[0].split(' / ') for row in rows]

summaries, licenses = zip(*summaries_and_licenses)



print(

    '{} package names, {} summaries, {} licenses'

    .format(len(package_names), len(summaries), len(licenses))

)
columns = {

    'package_name': package_names,

    'summary': summaries,

    'license': licenses

}

anaconda_metadata = pd.DataFrame(columns)

anaconda_metadata.set_index('package_name', inplace=True)

anaconda_metadata.head()
anaconda_metadata.to_csv('anaconda-package-metadata.csv', header=True)