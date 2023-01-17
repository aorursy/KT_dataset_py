!apt install python3-tk ghostscript -y
!pip install camelot-py[cv]
!pip install camelot-py[plot]
import camelot
!wget -O example.pdf http://apm.amegroups.com/article/download/38244/29000
tables = camelot.read_pdf('example.pdf', flavor='stream', pages='4', table_regions=["0,792,306,0"], row_tol=10, column_tol=0)

tables[0].df