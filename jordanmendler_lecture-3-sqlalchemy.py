import sqlalchemy



# Connect to the database...

from sqlalchemy import create_engine

engine = create_engine('sqlite:///:memory:', echo=False)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
from sqlalchemy import Column, Integer, String

class User(Base):

    __tablename__ = 'users'

    

    id = Column(Integer, primary_key=True)

    name = Column(String)

    fullname = Column(String)

    password = Column(String)

    

    # Defines to_string() representation 

    def __repr__(self):

        return "<User(name='%s', fullname='%s', password='%s')>" % (

                self.name, self.fullname, self.password)
Base.metadata.create_all(engine)
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)

session = Session()
ed_user = User(name='ed', fullname='Ed Jones', password='edspassword')

session.add(ed_user)

ed_user
our_user = session.query(User).filter_by(name='ed').first()
ed_user is our_user
session.add_all([

        User(name='wendy', fullname='Wendy Williams', password='foobar'),

        User(name='mary', fullname='Mary Contrary', password='xxg527'),

        User(name='fred', fullname='Fred Flinstone', password='blah')])



ed_user.password = 'f8s7ccs'
session.commit()
ed_user.name = 'Edwardo'



fake_user = User(name='fakeuser', fullname='Invalid', password='12345')

session.add(fake_user)



session.query(User).filter(User.name.in_(['Edwardo', 'fakeuser'])).all()
session.rollback()
ed_user.name
fake_user in session
for instance in session.query(User).order_by(User.id):

    print(instance)
for name, fullname in session.query(User.name, User.fullname):

    print(name, fullname)
for user in session.query(User).filter(User.name == 'ed'):

    print(user)
for user in session.query(User).filter(User.name != 'ed'):

    print(user)
for user in session.query(User).filter(User.name.like('%ed%')):

    print(user)
for user in session.query(User).filter(User.name.in_(['ed', 'wendy', 'jack'])):

    print(user)
for user in session.query(User).filter(~User.name.in_(['ed', 'wendy', 'jack'])):

    print(user)
for user in session.query(User).filter(User.name == 'ed', User.fullname == 'Ed Jones'):

    print(user)
from sqlalchemy import or_

for user in session.query(User).filter(or_(User.name == 'ed', User.name == 'wendy')):

    print(user)
from sqlalchemy import ForeignKey

from sqlalchemy.orm import relationship



class Address(Base):

    __tablename__ = 'addresses'

    id = Column(Integer, primary_key=True)

    email_address = Column(String, nullable=False)

    user_id = Column(Integer, ForeignKey('users.id'))

    user = relationship("User", back_populates="addresses")

    

    def __repr__(self):

        return "<Address(email_address='%s')>" % self.email_address
User.addresses = relationship("Address", order_by=Address.id, back_populates="user")



Base.metadata.create_all(engine) # Flush schema changes to the DBMS.
jack = User(name='jack', fullname='Jack Bean', password='gjffdd')

jack.addresses
jack.addresses = [Address(email_address='jack@google.com'), Address(email_address='j25@yahoo.com')]
jack.addresses[1]
jack.addresses[1].user
session.add(jack)



session.commit()



jack = session.query(User).filter_by(name='jack').one()

jack.addresses