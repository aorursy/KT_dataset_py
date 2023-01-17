! touch requirements.txt

! echo "dgl-cu101" > requirements.txt
! cat requirements.txt
! mkdir dgl

! pip download -r requirements.txt -d dgl
! ls dgl
! pip install --no-index --find-links ./dgl -r requirements.txt