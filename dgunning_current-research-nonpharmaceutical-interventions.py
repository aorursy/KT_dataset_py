from ipywidgets import Image

f = open("../input/cord-images/stayhome.png", "rb")

Image(value=f.read())
!pip install -U git+https://github.com/dgunning/cord19.git
import pandas as pd

from cord.core import image, get_docs

from cord import ResearchPapers

pd.options.display.max_colwidth=200




papers = ResearchPapers.load()
NON_PHARMACEUTICAL = r'.*(?<=non)[ \-]?pharmaceutical'

nonpharma_papers = papers.match(NON_PHARMACEUTICAL, column='abstract')

nonpharma_papers
nonpharma_papers.since_sarscov2()
papers[42305]
papers.similar_to('e4pr78n0')
from cord.tasks import Tasks



Tasks.NonPharmaceutical
papers.search_2d(Tasks.NonPharmaceutical[1].SeedQuestion)
papers.display('urkuofmu', '9bi6pobg', 'jtwb17u8', 'k4l45ene', 'e4pr78n0',  '48stbn6k', 'qqsefagq', 'lbcxl6w9', '1mpp7sbt', 'sgkuqcq6', 'zav5gksq', '0a49okho',

               'jmc9yhr0', 'g8lsrojl', 'z7r45291', 'zmqb140l', '01f5mvsc')
papers.search_2d(Tasks.NonPharmaceutical[2].SeedQuestion)
papers.display('28utunid', 'jab7vp33', 'ndscdqcb', 'k5q07y4b', 'jtwb17u8', 'aqwdg489', '901ghexi', 'kcb68hue', 'xfjexm5b','ln26bux8','0a49okho', "ngsstnpr", "gtfx5cp4",'qqsefagq', "tnw82hbm")
papers.similar_to('0a49okho')
papers.search_2d(Tasks.NonPharmaceutical[3].SeedQuestion)
image('../input/cord-images/cmmid_chart.png')
papers.display('jab7vp33','loi1vs5y','ass2u6y8', "oee19duz",'ls408b2b', "faec051u", "5gbkrs73", "q856rx6b",'ytsqfqvh', "jtwb17u8", "d4dnz2lk",'radi0wlh', "mzcajw8c", '8do4tojk')
papers.similar_to("d4dnz2lk")
papers.search_2d(Tasks.NonPharmaceutical[4].SeedQuestion)
papers.display('66ey5efz','y90w53xo', '2qthdldg', 'keaxietu','oribgtyl', 'w8ak5hpu', '0a49okho', '14x4uqq7', '71b6ai77', 'y8o5j2be')
papers.search_2d(Tasks.NonPharmaceutical[5].SeedQuestion)
papers.display('4kzv06e6','0z8x6v04', 's8fvqcel', 'ts3llerc', 'vwbpkpxd', '8f76vhyz', 'gv8wlo06', '6ymuovl2', '2h5oljm7', 'jljjqs6m')
papers.search_2d(Tasks.NonPharmaceutical[6].SeedQuestion)
papers.display("44inap6t",'2qthdldg', "w8ak5hpu", 'jdtmtjx5', '89xwnbbv', 'jvq1jtpz', 'dhy80rkn')
papers.search_2d(Tasks.NonPharmaceutical[7].SeedQuestion)
papers.display('5913rr94', '66ey5efz', "w8ak5hpu", "x1gces56", "ee53rfxw", 'kweh1doo', 'gj10u4lf')
papers.similar_to('66ey5efz')
papers.search_2d(Tasks.NonPharmaceutical[8].SeedQuestion)
papers.display('pd2palu4', '5xfgmi2n', '4kzv06e6', 'vhfgg4s4')
papers.similar_to('pd2palu4')
papers.display('qdamvwxl', '5wsj003j', 'ropgq7tr', '9skj0zbx', 'mqeg0oub', '844229sb', 'tfspedf1', 'b6zhp2ei')
get_docs('DesignNotes')
get_docs('SearchStrategy')
get_docs('Roadmap')
get_docs('References')